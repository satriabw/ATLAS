"""
Run from training/:  python -m unittest dataset.test_bev -v
"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from .bev import (BEVGrid, build_event_bev, CH_VEHICLE_COUNT, CH_PED_COUNT,
                  CH_VEHICLE_SPEED, CH_PED_SPEED, _to_loc, _to_frames)

PARQUET_DIR = Path(__file__).resolve().parents[2] / 'data/processed/interactions'
REAL_VIDEO, REAL_TRACK, REAL_ROI = 'video_001', 44, 'TOP'


def make_row(frames, v_xy, v_speed, p_track_id, p_xy, p_speed):
    """One parquet-shaped row: object arrays of per-frame values."""
    n = len(frames)
    return {
        'video_id': 'video_test',
        'frames': np.asarray(frames, dtype=np.int32),
        'v_track_id': 1,
        'v_loc_planar': np.array([np.asarray(p, dtype=np.float64) for p in v_xy],
                                 dtype=object),
        'v_speed': np.asarray(v_speed, dtype=np.float64),
        'p_track_id': p_track_id,
        'p_loc_planar': np.array([np.asarray(p, dtype=np.float64) for p in p_xy],
                                 dtype=object),
        'p_speed': np.asarray(p_speed, dtype=np.float64),
        'd_min': 1.0,
        'roi': 'TOP',
        '_n': n,
    }


def make_group(rows):
    return pd.DataFrame([{k: v for k, v in r.items() if k != '_n'} for r in rows])


class TestWorldToCell(unittest.TestCase):
    """T1 -- the world->cell transform, where a silent error still yields a
    plausible-looking map."""

    def setUp(self):
        self.grid = BEVGrid(x_min=-2.0, x_max=2.0, y_min=-1.0, y_max=1.0,
                            resolution=0.5)

    def test_lower_left_corner_is_origin_cell(self):
        # T1.1 anchors the transform: (x_min, y_min) must be cell (0, 0).
        r, c, v = self.grid.world_to_cell([[-2.0, -1.0]])
        self.assertTrue(v[0])
        self.assertEqual((r[0], c[0]), (0, 0))

    def test_interior_point_matches_hand_computed_index(self):
        # T1.2 catches sign flips and scale errors: x=-0.25 is 1.75 m from
        # x_min -> col 3; y=0.6 is 1.6 m from y_min -> row 3.
        r, c, v = self.grid.world_to_cell([[-0.25, 0.6]])
        self.assertTrue(v[0])
        self.assertEqual((r[0], c[0]), (3, 3))

    def test_cell_boundary_rounds_up_not_nearest(self):
        # T1.3 pins floor vs round. x_min+0.5 is exactly the 0|1 boundary and
        # must land in cell 1; a round() bug shifts the whole map half a cell.
        _, c, _ = self.grid.world_to_cell([[-1.5, -1.0]])
        self.assertEqual(c[0], 1)

    def test_out_of_extent_points_are_invalid_not_wrapped(self):
        # T1.4 the critical one: a negative numpy index wraps to the opposite
        # edge, so an off-grid pedestrian would appear as a phantom far away.
        pts = [[-2.5, 0.0], [2.0, 0.0], [0.0, -1.5], [0.0, 1.0]]
        _, _, v = self.grid.world_to_cell(pts)
        self.assertFalse(v.any(), 'points outside the half-open extent must be invalid')


class TestGridGeometry(unittest.TestCase):
    """T2 -- shape and axis order."""

    def test_dimensions_have_no_off_by_one(self):
        # T2.1 the default extent must give exactly the planned 75 x 64 grid.
        g = BEVGrid()
        self.assertEqual((g.H, g.W), (75, 64))

    def test_x_moves_column_and_y_moves_row(self):
        # T2.2 guards the (row=y, col=x) convention. An axis swap produces a
        # transposed map that still looks like a valid trajectory.
        g = BEVGrid(x_min=0.0, x_max=10.0, y_min=0.0, y_max=4.0, resolution=0.5)
        r0, c0, _ = g.world_to_cell([[0.25, 0.25]])
        rx, cx, _ = g.world_to_cell([[5.25, 0.25]])
        ry, cy, _ = g.world_to_cell([[0.25, 2.25]])
        self.assertEqual(r0[0], rx[0], '+x must not change the row')
        self.assertNotEqual(c0[0], cx[0], '+x must change the column')
        self.assertEqual(c0[0], cy[0], '+y must not change the column')
        self.assertNotEqual(r0[0], ry[0], '+y must change the row')


class TestChannelSemantics(unittest.TestCase):
    """T3 -- each channel means what the spec says."""

    def setUp(self):
        self.grid = BEVGrid(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0,
                            resolution=0.5)

    def test_single_vehicle_marks_exactly_one_cell(self):
        # T3.1 vehicle occupancy is a single cell of value 1 at the expected index.
        rows = [make_row([0], [(2.25, 3.25)], [4.0], 7, [(9.75, 9.75)], [1.0])]
        _, bev = build_event_bev(make_group(rows), self.grid)
        ch0 = bev[0, CH_VEHICLE_COUNT]
        self.assertEqual(ch0.sum(), 1.0)
        self.assertEqual(ch0[6, 4], 1.0)  # y=3.25 -> row 6, x=2.25 -> col 4

    def test_two_pedestrians_in_one_cell_accumulate(self):
        # T3.2 proves ch1 accumulates rather than overwrites. Measured on real
        # data: ~1% of occupied cells hold >=2 pedestrians at 0.5 m.
        rows = [
            make_row([0], [(0.25, 0.25)], [0.0], 7, [(5.10, 5.10)], [1.0]),
            make_row([0], [(0.25, 0.25)], [0.0], 8, [(5.40, 5.40)], [3.0]),
        ]
        _, bev = build_event_bev(make_group(rows), self.grid)
        self.assertEqual(bev[0, CH_PED_COUNT, 10, 10], 2.0)

    def test_pedestrian_speed_is_mean_not_sum(self):
        # T3.3 the behaviour explicitly requested: speeds 1.0 and 3.0 in the same
        # cell must give 2.0. A sum would give 4.0.
        rows = [
            make_row([0], [(0.25, 0.25)], [0.0], 7, [(5.10, 5.10)], [1.0]),
            make_row([0], [(0.25, 0.25)], [0.0], 8, [(5.40, 5.40)], [3.0]),
        ]
        _, bev = build_event_bev(make_group(rows), self.grid)
        self.assertAlmostEqual(float(bev[0, CH_PED_SPEED, 10, 10]), 2.0, places=6)

    def test_empty_cells_are_zero_without_nan(self):
        # T3.4 the mean divides by an occupancy count; unoccupied cells must not
        # produce 0/0 NaN.
        rows = [make_row([0], [(2.25, 3.25)], [4.0], 7, [(5.25, 5.25)], [1.0])]
        _, bev = build_event_bev(make_group(rows), self.grid)
        self.assertFalse(np.isnan(bev).any())
        self.assertEqual(bev[0, CH_PED_SPEED, 0, 0], 0.0)
        self.assertEqual(bev[0, CH_VEHICLE_SPEED, 0, 0], 0.0)

    def test_stopped_vehicle_is_occupied_with_zero_speed(self):
        # T3.5 documents the ambiguity: a stopped vehicle is indistinguishable
        # from empty in ch2 alone, so occupancy must be read from ch0.
        rows = [make_row([0], [(2.25, 3.25)], [0.0], 7, [(9.75, 9.75)], [1.0])]
        _, bev = build_event_bev(make_group(rows), self.grid)
        self.assertEqual(bev[0, CH_VEHICLE_COUNT, 6, 4], 1.0)
        self.assertEqual(bev[0, CH_VEHICLE_SPEED, 6, 4], 0.0)


class TestTemporal(unittest.TestCase):
    """T4 -- one map per timestep."""

    def setUp(self):
        self.grid = BEVGrid(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0,
                            resolution=0.5)
        self.rows = [
            make_row([10, 11, 12], [(1.25, 1.25), (2.25, 1.25), (3.25, 1.25)],
                     [5.0, 5.0, 5.0], 7,
                     [(6.25, 6.25), (6.25, 6.25), (6.25, 6.25)], [1.0, 1.0, 1.0]),
            make_row([11], [(2.25, 1.25)], [5.0], 8, [(8.25, 8.25)], [2.0]),
        ]

    def test_output_shape(self):
        # T4.1 shape contract (T, 4, H, W).
        frames, bev = build_event_bev(make_group(self.rows), self.grid)
        self.assertEqual(bev.shape, (3, 4, self.grid.H, self.grid.W))
        self.assertEqual(len(frames), 3)

    def test_single_frame_pedestrian_does_not_bleed(self):
        # T4.2 pedestrian 8 exists only at frame 11. If a buffer were reused
        # across timesteps every map would look identical.
        _, bev = build_event_bev(make_group(self.rows), self.grid)
        col = [bev[t, CH_PED_COUNT, 16, 16] for t in range(3)]
        self.assertEqual(col, [0.0, 1.0, 0.0])

    def test_frames_are_ascending_and_match_event(self):
        # T4.3 the time axis must be the event's own unique frames, in order.
        frames, _ = build_event_bev(make_group(self.rows), self.grid)
        np.testing.assert_array_equal(frames, [10, 11, 12])


@unittest.skipUnless((PARQUET_DIR / f'{REAL_VIDEO}_interactions.parquet').exists(),
                     'interaction parquets not available')
class TestRealEvent(unittest.TestCase):
    """T5 -- a real labelled violation event, which synthetic fixtures cannot
    exercise: parquet reassembly across 26 pedestrian rows and 344 frames."""

    @classmethod
    def setUpClass(cls):
        df = pd.read_parquet(PARQUET_DIR / f'{REAL_VIDEO}_interactions.parquet')
        cls.group = df[(df['v_track_id'] == REAL_TRACK) & (df['roi'] == REAL_ROI)]
        cls.grid = BEVGrid()
        cls.frames, cls.bev = build_event_bev(cls.group, cls.grid)

    def test_vehicle_present_exactly_once_per_timestep(self):
        # T5.1 the vehicle occupies every frame of its own event, and the frame
        # dedupe must leave exactly one observation -- a dedupe bug gives 2.0.
        per_t = self.bev[:, CH_VEHICLE_COUNT].sum(axis=(1, 2))
        np.testing.assert_allclose(per_t, 1.0)

    def test_pedestrian_mass_matches_observations_minus_clipped(self):
        # T5.2 no silent drops: rasterized pedestrian mass must equal the raw
        # observation count, less the out-of-extent points counted independently.
        expected = np.zeros(len(self.frames))
        clipped = np.zeros(len(self.frames))
        pos = {f: i for i, f in enumerate(self.frames)}
        for _, row in self.group.iterrows():
            f = _to_frames(row['frames'])
            _, _, valid = self.grid.world_to_cell(_to_loc(row['p_loc_planar']))
            for fi, ok in zip(f, valid):
                expected[pos[fi]] += 1
                if not ok:
                    clipped[pos[fi]] += 1
        np.testing.assert_allclose(self.bev[:, CH_PED_COUNT].sum(axis=(1, 2)),
                                   expected - clipped)
        self.assertGreater(expected.sum(), 0)

    def test_speed_channels_are_zero_where_unoccupied(self):
        # T5.3 speed must never leak into cells with no occupant.
        self.assertFalse((self.bev[:, CH_VEHICLE_SPEED][self.bev[:, CH_VEHICLE_COUNT] == 0] != 0).any())
        self.assertFalse((self.bev[:, CH_PED_SPEED][self.bev[:, CH_PED_COUNT] == 0] != 0).any())

    def test_vehicle_cells_match_independent_computation(self):
        # T5.4 end-to-end: the rasterized vehicle path must equal indices derived
        # straight from the parquet, bypassing the builder's reassembly.
        row = self.group.iloc[0]
        f = _to_frames(row['frames'])
        r, c, valid = self.grid.world_to_cell(_to_loc(row['v_loc_planar']))
        pos = {fr: i for i, fr in enumerate(self.frames)}
        for fi, ri, ci, ok in zip(f, r, c, valid):
            if ok:
                self.assertEqual(self.bev[pos[fi], CH_VEHICLE_COUNT, ri, ci], 1.0)

    def test_build_is_deterministic(self):
        # T6 guards against groupby/dict ordering leaking into the output.
        _, again = build_event_bev(self.group, self.grid)
        np.testing.assert_array_equal(self.bev, again)


if __name__ == '__main__':
    unittest.main()
