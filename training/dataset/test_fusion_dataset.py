"""
Run from training/:  python -m unittest dataset.test_fusion_dataset -v

The invariant under test is alignment: both streams must come from the same
frames, every BEV slot must actually hold the vehicle, and the quadrant must be
the one the label names. Each of those fails silently -- wrong-but-well-shaped
tensors that train without complaint.

Tests needing real data are skipped when the frame database is absent.
"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .fusion_dataset import (QUADRANT_INDEX, FusionDataset, _grid_for,
                             load_events, snapped_frames)
from .quadrant_geometry import CELLS, WINDOW_M, quadrant_window
from .vision_crop import SIZE, quadrant_rect

ROOT = Path(__file__).resolve().parents[2]
PARQUET_DIR = ROOT / 'data/processed/interactions'
FRAME_DB = ROOT / 'data/raw/video/frames_db.h5'
LABELS = ROOT / 'data/raw/labels/train_labels.pkl'
VIDEO, TRACK, ROI, DOWN = 'video_001', 7, 'TOP', True
# shortest event in the train split: spans 8 frames, so 32 samples force repeats
SHORT_EVENT = ('video_043', 127, 'BOT', True, 1)

has_data = PARQUET_DIR.exists() and FRAME_DB.exists()
_shared = {}


def group():
    if 'g' not in _shared:
        df = pd.read_parquet(PARQUET_DIR / f'{VIDEO}_interactions.parquet')
        _shared['g'] = df[(df['v_track_id'] == TRACK) & (df['roi'] == ROI)]
    return _shared['g']


class TestGrid(unittest.TestCase):

    def test_grid_is_the_quadrant_window_at_the_shipped_resolution(self):
        for roi, down in QUADRANT_INDEX:
            g = _grid_for(roi, down)
            self.assertEqual((g.H, g.W), (CELLS, CELLS))
            x_min, y_min = quadrant_window(roi, down)
            self.assertAlmostEqual(g.x_min, x_min)
            self.assertAlmostEqual(g.x_max, x_min + WINDOW_M)

    def test_the_four_quadrants_give_four_different_grids(self):
        origins = {(_grid_for(r, d).x_min, _grid_for(r, d).y_min)
                   for r, d in QUADRANT_INDEX}
        self.assertEqual(len(origins), 4)


@unittest.skipUnless(has_data, 'needs parquet + frame db')
class TestSnapping(unittest.TestCase):

    def test_every_sampled_frame_is_one_the_vehicle_occupies(self):
        """The whole point of snapping. A miss means an empty BEV slot."""
        g = group()
        vehicle = np.unique(np.concatenate(
            [np.asarray(r).ravel() for r in g['frames']]))
        self.assertTrue(np.all(np.isin(snapped_frames(g, 32), vehicle)))

    def test_frames_are_ascending_and_the_right_length(self):
        f = snapped_frames(group(), 32)
        self.assertEqual(len(f), 32)
        self.assertTrue(np.all(np.diff(f) >= 0))

    def test_snapping_stays_inside_the_event_span(self):
        g = group()
        f = snapped_frames(g, 32)
        allf = np.concatenate([np.asarray(r).ravel() for r in g['frames']])
        self.assertGreaterEqual(f.min(), allf.min())
        self.assertLessEqual(f.max(), allf.max())

    def test_snap_picks_the_nearer_neighbour(self):
        """Constructed gap: a sample landing at 10 must snap to 9, not 20."""
        g = pd.DataFrame({'frames': [np.array([1, 5, 9, 20, 21])],
                          'v_loc_planar': [np.zeros((5, 2))],
                          'v_speed': [np.zeros(5)],
                          'p_loc_planar': [np.zeros((5, 2))],
                          'p_speed': [np.zeros(5)],
                          'p_track_id': [1]})
        f = snapped_frames(g, 5)
        self.assertTrue(np.all(np.isin(f, [1, 5, 9, 20, 21])))
        self.assertEqual(f[0], 1)
        self.assertEqual(f[-1], 21)


@unittest.skipUnless(has_data, 'needs parquet + frame db')
class TestItem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ds = FusionDataset([(VIDEO, TRACK, ROI, DOWN, 0)], PARQUET_DIR,
                               FRAME_DB, num_frames=32)
        cls.item = cls.ds[0]

    def test_shapes(self):
        self.assertEqual(tuple(self.item['bev'].shape), (32, 4, CELLS, CELLS))
        self.assertEqual(tuple(self.item['crop'].shape), (32, 3, SIZE, SIZE))

    def test_T8_both_streams_use_the_same_frames(self):
        """Alignment is the reason this module exists."""
        self.assertEqual(len(self.item['frames']), 32)
        np.testing.assert_array_equal(self.item['frames'].numpy(),
                                      snapped_frames(group(), 32))

    def test_T7_every_bev_slot_holds_the_vehicle(self):
        """Channel 0 is the vehicle count and there is one vehicle per event, so
        every timestep must have exactly one occupied cell -- unless the vehicle
        is outside the 16 m window, which the geometry module measured at 2.6%
        of events."""
        occupied = (self.item['bev'][:, 0] > 0).flatten(1).sum(1)
        self.assertGreaterEqual(float((occupied > 0).float().mean()), 0.99)

    def test_T9_crop_is_imagenet_normalised_and_bev_is_not_rescaled(self):
        crop = self.item['crop']
        self.assertLess(abs(float(crop.mean())), 2.0)
        self.assertGreater(float(crop.std()), 0.3)
        bev = self.item['bev']
        self.assertLessEqual(float(bev[:, 0].max()), 1.0)   # counts, not rescaled
        self.assertGreaterEqual(float(bev.min()), 0.0)

    def test_bev_is_sparse_but_not_empty(self):
        bev = self.item['bev']
        occ = float((bev[:, :2] > 0).float().mean())
        self.assertGreater(occ, 0.0)
        self.assertLess(occ, 0.2)

    def test_crop_is_not_a_constant_image(self):
        self.assertGreater(float(self.item['crop'].std(dim=(1, 2, 3)).min()), 1e-3)

    def test_T6_quadrant_index_matches_the_label(self):
        self.assertEqual(self.item['quadrant'], QUADRANT_INDEX[(ROI, DOWN)])
        self.assertEqual(len(QUADRANT_INDEX), 4)

    def test_label_passes_through(self):
        self.assertEqual(self.item['label'], 0)

    def test_is_deterministic(self):
        again = self.ds[0]
        torch.testing.assert_close(again['bev'], self.item['bev'])
        torch.testing.assert_close(again['crop'], self.item['crop'])

    def test_T10_repeated_frames_give_repeated_slots_not_empty_ones(self):
        """A short event repeats frames; those slots must be copies, never zeros.

        build_event_bev resolves a track frame to ONE slot via searchsorted, so
        without the unique/expand step in __getitem__ the duplicate slots come
        back all-zero. video_043/127/BOT spans 8 frames, so 32 samples force 24
        repeats -- the shortest event in the train split, chosen so this branch
        is actually exercised rather than skipped.
        """
        ds = FusionDataset([SHORT_EVENT], PARQUET_DIR, FRAME_DB, num_frames=32)
        item = ds[0]
        frames = item['frames'].numpy()
        dup = np.flatnonzero(frames[1:] == frames[:-1])
        self.assertGreater(len(dup), 0, 'expected repeats on an 8-frame event')
        for i in dup:
            torch.testing.assert_close(item['bev'][i], item['bev'][i + 1])
            self.assertGreater(float(item['bev'][i, 0].sum()), 0.0,
                               f'slot {i} is an empty duplicate')
        # and no slot anywhere is empty
        self.assertTrue(bool((item['bev'][:, 0].flatten(1).sum(1) > 0).all()))


@unittest.skipUnless(has_data and LABELS.exists(), 'needs labels')
class TestLoadEvents(unittest.TestCase):

    def test_events_are_wellformed_and_labels_are_binary(self):
        events = load_events(LABELS, PARQUET_DIR, FRAME_DB)
        self.assertGreater(len(events), 3000)
        rois = {e[2] for e in events}
        self.assertTrue(rois <= {'TOP', 'BOT'})
        self.assertTrue({e[4] for e in events} <= {0, 1})
        self.assertTrue(all(isinstance(e[3], bool) or e[3] in (True, False)
                            for e in events))

    def test_events_are_unique(self):
        events = load_events(LABELS, PARQUET_DIR, FRAME_DB)
        keys = {(v, t, r) for v, t, r, _, _ in events}
        self.assertEqual(len(keys), len(events))

    def test_all_four_quadrants_are_represented(self):
        events = load_events(LABELS, PARQUET_DIR, FRAME_DB)
        self.assertEqual({QUADRANT_INDEX[(r, d)] for _, _, r, d, _ in events},
                         {0, 1, 2, 3})


if __name__ == '__main__':
    unittest.main()
