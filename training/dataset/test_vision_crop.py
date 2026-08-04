"""
Run from training/:  python -m unittest dataset.test_vision_crop -v

The crop is the whole grounding mechanism -- picking the rectangle IS the ROI and
direction conditioning -- so the failure modes are silent and expensive: a
transposed rectangle, a BGR clip fed to an RGB-normalized trunk, or a drift away
from the Crosswalk reference all produce a perfectly well-formed tensor.
"""
import re
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .vision_crop import (CROP_LB, CROP_LT, CROP_RB, CROP_RT, IMAGENET_MEAN,
                          IMAGENET_STD, SIZE, crop_clip, event_frame_grid,
                          imagenet_denormalize, parse_label, quadrant_rect)

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'crosswalk-original/preprocessing_vr.py'
PARQUET_DIR = ROOT / 'data/processed/interactions'
FRAME_DB = ROOT / 'data/raw/video/frames_db.h5'


class FakeFrames:
    """Stands in for an h5 dataset of JPEG-encoded frames."""

    def __init__(self, images):
        self._enc = [cv2.imencode('.png', im)[1] for im in images]
        self.shape = (len(images),)

    def __getitem__(self, i):
        return self._enc[i]


class TestLabelParsing(unittest.TestCase):

    def test_parses_a_known_label(self):
        self.assertEqual(parse_label('V001I00002S1D0R0A1'),
                         ('video_001', 2, 'BOT', True, 1))

    def test_roi_bit_maps_s0_to_top_and_s1_to_bot(self):
        self.assertEqual(parse_label('V007I00042S0D1R0A0')[2], 'TOP')
        self.assertEqual(parse_label('V007I00042S1D1R0A0')[2], 'BOT')

    def test_direction_bit_d0_is_downward(self):
        self.assertTrue(parse_label('V007I00042S0D0R0A0')[3])
        self.assertFalse(parse_label('V007I00042S0D1R0A0')[3])

    def test_annotation_bit_is_last(self):
        self.assertEqual(parse_label('V007I00042S0D0R0A0')[4], 0)   # violation
        self.assertEqual(parse_label('V007I00042S0D0R0A1')[4], 1)   # compliance

    def test_video_id_is_zero_padded_to_three(self):
        self.assertEqual(parse_label('V120I00001S0D0R0A0')[0], 'video_120')

    def test_garbage_is_rejected_not_silently_parsed(self):
        for bad in ('', 'nonsense', 'V001I2S1D0R0'):
            with self.assertRaises(ValueError):
                parse_label(bad)


class TestQuadrantSelection(unittest.TestCase):
    """The rectangle IS the ROI x direction conditioning."""

    def test_all_four_combinations_are_distinct(self):
        rects = {quadrant_rect(r, d) for r in ('TOP', 'BOT') for d in (True, False)}
        self.assertEqual(len(rects), 4)

    def test_selection_matches_the_reference_mapping(self):
        self.assertEqual(quadrant_rect('TOP', True), CROP_RT)
        self.assertEqual(quadrant_rect('TOP', False), CROP_LT)
        self.assertEqual(quadrant_rect('BOT', True), CROP_RB)
        self.assertEqual(quadrant_rect('BOT', False), CROP_LB)

    def test_bad_roi_is_rejected(self):
        with self.assertRaises(ValueError):
            quadrant_rect('MIDDLE', True)

    def test_rectangles_are_well_formed_and_inside_the_frame(self):
        for rect in (CROP_LB, CROP_RB, CROP_LT, CROP_RT):
            x0, y0, x1, y1 = rect
            self.assertLess(x0, x1)
            self.assertLess(y0, y1)
            self.assertGreaterEqual(min(x0, y0), 0)
            self.assertLessEqual(x1, 1200)
            self.assertLessEqual(y1, 1100)

    @unittest.skipUnless(REFERENCE.exists(), 'Crosswalk reference not available')
    def test_rectangles_still_match_the_crosswalk_source(self):
        """vision_crop.py claims these are verbatim; this makes that falsifiable."""
        src = REFERENCE.read_text()
        for name, ours in (('crop_lb', CROP_LB), ('crop_rb', CROP_RB),
                           ('crop_lt', CROP_LT), ('crop_rt', CROP_RT)):
            m = re.search(rf'^{name}\s*=\s*\(([^)]*)\)', src, re.M)
            self.assertIsNotNone(m, f'{name} not found in the reference')
            theirs = tuple(int(v) for v in m.group(1).split(','))
            self.assertEqual(ours, theirs, f'{name} drifted from Crosswalk')


class TestFrameGrid(unittest.TestCase):

    def _group(self, frames):
        return pd.DataFrame({'frames': [np.array(frames)]})

    def test_grid_has_the_requested_length(self):
        self.assertEqual(len(event_frame_grid(self._group(range(100)), 32)), 32)

    def test_grid_is_ascending(self):
        g = event_frame_grid(self._group(range(500)), 32)
        self.assertTrue(np.all(np.diff(g) >= 0))

    def test_grid_spans_the_whole_event(self):
        g = event_frame_grid(self._group(range(10, 210)), 16)
        self.assertEqual((g[0], g[-1]), (10, 209))

    def test_short_event_repeats_frames_rather_than_padding(self):
        """A padded clip would feed the trunk black frames; repeats are honest."""
        g = event_frame_grid(self._group([5, 6, 7]), 8)
        self.assertEqual(len(g), 8)
        self.assertEqual((g.min(), g.max()), (5, 7))
        self.assertLess(len(np.unique(g)), 8)

    def test_multi_row_group_is_spanned_across_all_rows(self):
        df = pd.DataFrame({'frames': [np.array([10, 11]), np.array([80, 81])]})
        g = event_frame_grid(df, 4)
        self.assertEqual((g[0], g[-1]), (10, 81))


class TestCropClip(unittest.TestCase):

    def _frames(self, n=3, colour=(0, 0, 255)):     # BGR: pure red
        img = np.zeros((1100, 1200, 3), np.uint8)
        img[:, :] = colour
        return FakeFrames([img] * n)

    def test_shape_and_dtype(self):
        clip = crop_clip(self._frames(), np.array([1, 2, 3]), CROP_RT)
        self.assertEqual(clip.shape, (3, 3, SIZE, SIZE))
        self.assertEqual(clip.dtype, np.float32)

    def test_channels_are_rgb_not_bgr(self):
        """The trunk is ImageNet-normalized in RGB; feeding BGR is silent."""
        clip = crop_clip(self._frames(colour=(0, 0, 255)), np.array([1]), CROP_RT,
                         normalize=False)
        r, g, b = clip[0].mean(axis=(1, 2))
        self.assertGreater(r, 0.9)
        self.assertLess(max(g, b), 0.1)

    def test_unnormalized_output_is_in_zero_one(self):
        clip = crop_clip(self._frames(), np.array([1]), CROP_LT, normalize=False)
        self.assertGreaterEqual(clip.min(), 0.0)
        self.assertLessEqual(clip.max(), 1.0)

    def test_normalization_applies_imagenet_statistics(self):
        raw = crop_clip(self._frames(), np.array([1]), CROP_LT, normalize=False)
        norm = crop_clip(self._frames(), np.array([1]), CROP_LT, normalize=True)
        expect = (raw[0, 0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        np.testing.assert_allclose(norm[0, 0], expect, rtol=1e-5)

    def test_denormalize_round_trips(self):
        norm = crop_clip(self._frames(), np.array([1]), CROP_LT, normalize=True)
        back = imagenet_denormalize(norm)
        np.testing.assert_allclose(back[0, :, :, 0], 1.0, atol=1e-4)

    def test_out_of_range_frame_numbers_are_clamped_not_wrapped(self):
        """A negative index would silently sample the end of the video."""
        clip = crop_clip(self._frames(n=3), np.array([0, 999]), CROP_LT)
        self.assertTrue(np.isfinite(clip).all())
        self.assertEqual(clip.shape[0], 2)

    def test_different_rectangles_see_different_pixels(self):
        img = np.zeros((1100, 1200, 3), np.uint8)
        img[50:470, 20:580] = (255, 255, 255)          # only the LT quadrant
        ds = FakeFrames([img])
        lt = crop_clip(ds, np.array([1]), CROP_LT, normalize=False)
        rt = crop_clip(ds, np.array([1]), CROP_RT, normalize=False)
        self.assertGreater(lt.mean(), 0.9)
        self.assertLess(rt.mean(), 0.5)


@unittest.skipUnless(FRAME_DB.exists(), 'frame database not available')
class TestRealEventIntegration(unittest.TestCase):

    def test_crops_a_real_event_to_a_usable_clip(self):
        import h5py
        vid, tid, roi, down = 'video_001', 7, 'TOP', True
        df = pd.read_parquet(PARQUET_DIR / f'{vid}_interactions.parquet')
        g = df[(df['v_track_id'] == tid) & (df['roi'] == roi)]
        grid = event_frame_grid(g, 4)
        with h5py.File(FRAME_DB, 'r') as db:
            clip = crop_clip(db[vid], grid, quadrant_rect(roi, down))
        self.assertEqual(clip.shape, (4, 3, SIZE, SIZE))
        self.assertTrue(np.isfinite(clip).all())
        self.assertGreater(clip.std(), 0.3, 'real crop is nearly constant')


if __name__ == '__main__':
    unittest.main()
