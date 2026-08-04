"""
Run from training/:  python -m unittest dataset.test_quadrant_geometry -v

The failure modes here are silent. A transposed or flipped projection matrix
still produces a well-shaped M that trains without complaint and is simply
wrong; a column that spreads downward instead of upward means the 3x4 matrix
is not what we think it is; an M that is identical across quadrants means the
quadrant argument is being ignored somewhere. Each of those has a test.
"""
import unittest

import numpy as np

from .quadrant_geometry import (CELLS, HEIGHT_M, RESOLUTION, VISION_GRID, WINDOW_M,
                                Z_UP, cell_centres, correspondence, image_to_world,
                                quadrant_window, world_to_image)
from .vision_crop import quadrant_rect

QUADS = [('TOP', True), ('TOP', False), ('BOT', True), ('BOT', False)]
GRID = 8


def _ground_only_M(roi, downward, grid=GRID):
    """M built from the ground plane alone, for comparison against the column."""
    x0, y0, x1, y1 = quadrant_rect(roi, downward)
    centres = cell_centres(roi, downward)
    uv = world_to_image(np.hstack([centres, np.zeros((len(centres), 1))]))
    block = CELLS // grid
    rows = np.repeat(np.arange(grid), block)
    bev_tok = (rows[:, None] * grid + rows[None, :]).ravel()
    M = np.zeros((grid * grid, VISION_GRID ** 2), dtype=np.float64)
    inside = (uv[:, 0] >= x0) & (uv[:, 0] < x1) & (uv[:, 1] >= y0) & (uv[:, 1] < y1)
    tu = np.clip(((uv[inside, 0] - x0) / (x1 - x0) * VISION_GRID).astype(int), 0, VISION_GRID - 1)
    tv = np.clip(((uv[inside, 1] - y0) / (y1 - y0) * VISION_GRID).astype(int), 0, VISION_GRID - 1)
    np.add.at(M, (bev_tok[inside], tv * VISION_GRID + tu), 1.0)
    return M


def _connected(vision_tokens):
    """Are these 7x7 token indices 4-connected?"""
    cells = {(int(k) // VISION_GRID, int(k) % VISION_GRID) for k in vision_tokens}
    if not cells:
        return True
    seen, stack = set(), [next(iter(cells))]
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if nb in cells and nb not in seen:
                stack.append(nb)
    return seen == cells


class TestProjection(unittest.TestCase):
    """T1 -- the calibration round-trips. Everything else rests on this."""

    def test_image_to_world_to_image_returns_the_pixel(self):
        uv = np.array([[300., 300.], [600., 500.], [900., 200.], [150., 700.]])
        back = world_to_image(np.hstack([image_to_world(uv), np.zeros((len(uv), 1))]))
        np.testing.assert_allclose(back, uv, atol=1e-6)

    def test_world_to_image_to_world_returns_the_point(self):
        xy = np.array([[0., 0.], [-5., 3.], [8., -2.], [2.5, 10.]])
        uv = world_to_image(np.hstack([xy, np.zeros((len(xy), 1))]))
        np.testing.assert_allclose(image_to_world(uv), xy, atol=1e-6)

    def test_raising_z_moves_the_pixel_up_not_down(self):
        """A point above the ground must appear HIGHER in the image (smaller v).

        Up is -Z in this calibration (camera centre at world Z = -89.5). Getting
        the sign backwards is silent: the column keeps its length and M keeps
        its shape, it just reaches into the road instead of the object. This
        test caught exactly that on the first run.
        """
        xy = cell_centres('TOP', True)[::37]
        ground = world_to_image(np.hstack([xy, np.zeros((len(xy), 1))]))
        raised = world_to_image(np.hstack([xy, np.full((len(xy), 1), Z_UP * 1.5)]))
        self.assertTrue(np.all(raised[:, 1] < ground[:, 1]))
        # and the wrong sign must fail, so the test cannot pass vacuously
        wrong = world_to_image(np.hstack([xy, np.full((len(xy), 1), -Z_UP * 1.5)]))
        self.assertTrue(np.all(wrong[:, 1] > ground[:, 1]))

    def test_the_measured_height_offset_is_about_one_vision_token(self):
        """1.5 m -> ~70 px, and a quadrant crop is ~60-81 px per vision token."""
        xy = cell_centres('TOP', True)[::37]
        ground = world_to_image(np.hstack([xy, np.zeros((len(xy), 1))]))
        raised = world_to_image(np.hstack([xy, np.full((len(xy), 1), Z_UP * 1.5)]))
        dv = np.abs(raised[:, 1] - ground[:, 1])
        self.assertTrue(np.all((dv > 60) & (dv < 80)), f'offsets {dv}')


class TestWindow(unittest.TestCase):

    def test_window_is_centred_on_the_crop_footprint(self):
        for roi, down in QUADS:
            x0, y0, x1, y1 = quadrant_rect(roi, down)
            u, v = np.meshgrid(np.linspace(x0, x1, 40), np.linspace(y0, y1, 40))
            w = image_to_world(np.stack([u.ravel(), v.ravel()], axis=1))
            cx = (w[:, 0].min() + w[:, 0].max()) / 2
            cy = (w[:, 1].min() + w[:, 1].max()) / 2
            wx, wy = quadrant_window(roi, down)
            self.assertAlmostEqual(wx + WINDOW_M / 2, cx, places=6)
            self.assertAlmostEqual(wy + WINDOW_M / 2, cy, places=6)

    def test_the_four_windows_are_distinct(self):
        origins = {quadrant_window(r, d) for r, d in QUADS}
        self.assertEqual(len(origins), 4)

    def test_cell_centres_are_row_major_with_row_axis_y(self):
        """Must match BEVGrid's convention or the raster and M disagree."""
        c = cell_centres('BOT', True)
        self.assertEqual(c.shape, (CELLS * CELLS, 2))
        self.assertAlmostEqual(c[1, 0] - c[0, 0], RESOLUTION)   # along a row: x moves
        self.assertAlmostEqual(c[1, 1] - c[0, 1], 0.0)
        self.assertAlmostEqual(c[CELLS, 1] - c[0, 1], RESOLUTION)  # next row: y moves
        self.assertAlmostEqual(c[CELLS, 0] - c[0, 0], 0.0)

    def test_cells_span_exactly_the_window(self):
        c = cell_centres('TOP', False)
        wx, wy = quadrant_window('TOP', False)
        self.assertAlmostEqual(c[:, 0].min() - RESOLUTION / 2, wx)
        self.assertAlmostEqual(c[:, 0].max() + RESOLUTION / 2, wx + WINDOW_M)


class TestCorrespondence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.M = {q: correspondence(*q, GRID) for q in QUADS}

    def test_shape(self):
        for q in QUADS:
            self.assertEqual(self.M[q].shape, (GRID * GRID, VISION_GRID ** 2))

    def test_T2_rows_sum_to_one_or_exactly_zero(self):
        for q in QUADS:
            s = self.M[q].sum(axis=1)
            paired = s > 0
            np.testing.assert_allclose(s[paired], 1.0, atol=1e-6)
            self.assertTrue(np.all(s[~paired] == 0.0))

    def test_T3_paired_fraction_matches_the_measurement(self):
        """Measured 70.3-75.0%. Not 0 (nothing maps) and not 1 (window too small)."""
        for q in QUADS:
            frac = (self.M[q].sum(axis=1) > 0).mean()
            self.assertGreater(frac, 0.65)
            self.assertLess(frac, 0.85)

    def test_T4_each_token_maps_to_a_contiguous_vision_region(self):
        """A scrambled projection yields scattered footprints and passes shape
        checks; contiguity is what actually catches it."""
        for q in QUADS:
            for i in np.flatnonzero(self.M[q].sum(axis=1) > 0):
                self.assertTrue(_connected(np.flatnonzero(self.M[q][i])),
                                f'{q} token {i} maps to a disconnected region')

    def test_T5_the_four_quadrants_give_four_different_M(self):
        for i in range(len(QUADS)):
            for j in range(i + 1, len(QUADS)):
                self.assertFalse(np.allclose(self.M[QUADS[i]], self.M[QUADS[j]]),
                                 f'{QUADS[i]} and {QUADS[j]} share an M')

    def test_T5b_column_covers_the_ground_mapping_and_extends_upward(self):
        """Every vision token the ground mapping reaches must still be reached,
        and every token the column ADDS must lie above it (smaller row)."""
        for q in QUADS:
            ground = _ground_only_M(*q, GRID)
            for i in np.flatnonzero(ground.sum(axis=1) > 0):
                g = set(np.flatnonzero(ground[i]).tolist())
                c = set(np.flatnonzero(self.M[q][i]).tolist())
                self.assertTrue(g <= c, f'{q} token {i}: column dropped ground tokens')
                g_rows = [k // VISION_GRID for k in g]
                for k in c - g:
                    self.assertLessEqual(k // VISION_GRID, max(g_rows),
                                         f'{q} token {i}: column reached downward')

    def test_T5c_vertical_span_matches_the_measurement(self):
        """Measured 2.56-2.79 rows mean, max 4. The column contributes ~1.3 and
        the token's own 2 m ground depth the rest. A span near 7 means the z
        range or the scaling is wrong."""
        for q in QUADS:
            spans = []
            for i in np.flatnonzero(self.M[q].sum(axis=1) > 0):
                rows = np.flatnonzero(self.M[q][i]) // VISION_GRID
                spans.append(rows.max() - rows.min() + 1)
            self.assertGreater(np.mean(spans), 2.0)
            self.assertLess(np.mean(spans), 3.5)
            self.assertLessEqual(max(spans), 5)

    def test_fan_out_is_small_enough_to_be_local(self):
        """Measured 4.79-5.33 of 49. A fan-out near 49 is a uniform blur, i.e.
        no correspondence at all."""
        for q in QUADS:
            fan = [(self.M[q][i] > 0).sum()
                   for i in np.flatnonzero(self.M[q].sum(axis=1) > 0)]
            self.assertLess(np.mean(fan), 8.0)

    def test_neighbouring_bev_tokens_map_to_neighbouring_vision_regions(self):
        """Adjacent in the world must stay adjacent in the image -- the property
        a rotated or transposed mapping breaks."""
        for q in QUADS:
            M = self.M[q]
            for r in range(GRID):
                for c in range(GRID - 1):
                    a, b = M[r * GRID + c], M[r * GRID + c + 1]
                    if a.sum() == 0 or b.sum() == 0:
                        continue
                    ca = np.array([np.flatnonzero(a) // VISION_GRID,
                                   np.flatnonzero(a) % VISION_GRID]).mean(axis=1)
                    cb = np.array([np.flatnonzero(b) // VISION_GRID,
                                   np.flatnonzero(b) % VISION_GRID]).mean(axis=1)
                    self.assertLess(np.hypot(*(ca - cb)), 3.0)

    def test_rejects_a_grid_that_does_not_divide_the_cells(self):
        with self.assertRaises(ValueError):
            correspondence('TOP', True, 7)

    def test_is_deterministic(self):
        np.testing.assert_array_equal(correspondence('TOP', True, GRID),
                                      correspondence('TOP', True, GRID))


if __name__ == '__main__':
    unittest.main()
