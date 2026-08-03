"""
Run from training/:  python -m unittest models.test_bev_encoder -v

The encoder consumes flattened BEV maps (B*T, 4, 75, 64) and returns a 5x4 grid
of 20 spatial tokens per map. Most failure modes here are silent: a dead channel, a collapsed
embedding, a scrambled token order, or a 5D tensor reinterpreted as something
else all still produce a tensor of plausible shape.
"""
import unittest
from pathlib import Path

import numpy as np
import torch

from .bev_encoder import BEVEncoder, TOKEN_H, TOKEN_W
from dataset.bev import BEVGrid, build_event_bev, NUM_CHANNELS

PARQUET_DIR = Path(__file__).resolve().parents[2] / 'data/processed/interactions'
REAL_VIDEO, REAL_TRACK, REAL_ROI = 'video_001', 44, 'TOP'

H, W = 75, 64


def blob(row, col, channel=0, value=1.0, h=H, w=W, channels=NUM_CHANNELS):
    """A single-map BEV with one occupied cell -- the sparsity of a real map."""
    x = torch.zeros(1, channels, h, w)
    x[0, channel, row, col] = value
    return x


def rel_diff(a, b):
    """L2 distance between two embeddings, scaled by their typical norm."""
    a, b = a.detach(), b.detach()
    scale = (a.norm() + b.norm()) / 2
    return float((a - b).norm() / scale.clamp(min=1e-8))


class TestOutputContract(unittest.TestCase):
    """E1 -- shape and dtype, the part a caller wires into a fusion model."""

    def setUp(self):
        torch.manual_seed(0)
        self.enc = BEVEncoder().eval()

    def test_output_shape_is_batch_by_tokens_by_d_out(self):
        # E1.1 the contract the fused branch consumes: one sequence of 20 spatial
        # tokens per map, ready to concatenate with vision tokens.
        out = self.enc(torch.randn(6, NUM_CHANNELS, H, W))
        self.assertEqual(out.shape, (6, TOKEN_H * TOKEN_W, 128))

    def test_d_out_and_in_channels_are_honoured(self):
        # E1.2 constructor args must reach the layers, not be decorative.
        enc = BEVEncoder(in_channels=2, d_out=32).eval()
        self.assertEqual(enc(torch.randn(3, 2, H, W)).shape, (3, 20, 32))

    def test_batch_of_one_works(self):
        # E1.3 a single-timestep event must work in both modes. GroupNorm has no
        # batch-size floor, unlike the BatchNorm this replaced.
        enc = BEVEncoder()
        self.assertEqual(enc.eval()(torch.randn(1, NUM_CHANNELS, H, W)).shape, (1, 20, 128))
        self.assertEqual(enc.train()(torch.randn(1, NUM_CHANNELS, H, W)).shape, (1, 20, 128))

    def test_eval_mode_is_deterministic(self):
        # E1.4 no dropout/randomness: two identical calls must match bit-for-bit.
        x = torch.randn(4, NUM_CHANNELS, H, W)
        torch.testing.assert_close(self.enc(x), self.enc(x), rtol=0, atol=0)

    def test_eval_mode_samples_are_independent(self):
        # E1.5 a map's tokens must not depend on what else is in the batch, or an
        # event's features change with batch composition and evaluation is not
        # reproducible. E6.1 pins the same property in train mode, where the
        # BatchNorm this replaced actually broke it.
        x = torch.randn(4, NUM_CHANNELS, H, W)
        alone = self.enc(x[:1])
        batched = self.enc(x)[:1]
        torch.testing.assert_close(alone, batched)


class TestNumericalHealth(unittest.TestCase):
    """E2 -- finiteness and gradient flow on inputs the real data produces."""

    def setUp(self):
        torch.manual_seed(0)
        self.enc = BEVEncoder()

    def test_empty_grid_gives_finite_output(self):
        # E2.1 an all-zero map is legal (a timestep whose points fall off-extent).
        out = self.enc.eval()(torch.zeros(2, NUM_CHANNELS, H, W))
        self.assertTrue(torch.isfinite(out).all())

    def test_sparse_realistic_input_is_finite(self):
        # E2.2 real maps are ~0.2% occupied with speeds averaging 0.9/0.7 where
        # occupied; 20.0 here is far above the measured 5.6/7.8 maximum, so this
        # is the out-of-range case rather than the typical one.
        x = torch.zeros(4, NUM_CHANNELS, H, W)
        idx = torch.randint(0, H * W, (4, 5))
        for b in range(4):
            for k in idx[b]:
                x[b, 0, k // W, k % W] = 1.0
                x[b, 2, k // W, k % W] = 20.0
        self.assertTrue(torch.isfinite(self.enc.eval()(x)).all())

    def test_every_parameter_receives_gradient(self):
        # E2.3 catches a layer detached from the output path -- it would train
        # forever with that block frozen at its random init.
        out = self.enc.train()(torch.randn(8, NUM_CHANNELS, H, W))
        out.sum().backward()
        for name, p in self.enc.named_parameters():
            self.assertIsNotNone(p.grad, f'{name} has no grad')
            self.assertTrue(torch.isfinite(p.grad).all(), f'{name} grad not finite')
            if 'bias' not in name:
                self.assertGreater(float(p.grad.abs().sum()), 0.0, f'{name} grad is all zero')

    def test_gradient_reaches_the_input_map(self):
        # E2.4 required for end-to-end training against a BEV builder upstream,
        # and proves nothing in the stack blocks backprop.
        x = torch.randn(4, NUM_CHANNELS, H, W, requires_grad=True)
        self.enc.train()(x).sum().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0.0)


class TestFeatureSensitivity(unittest.TestCase):
    """E3 -- the embedding must actually depend on the map. These are the tests
    that separate 'runs' from 'extracts features'."""

    def setUp(self):
        torch.manual_seed(0)
        self.enc = BEVEncoder().eval()
        self.empty = self.enc(torch.zeros(1, NUM_CHANNELS, H, W)).detach()

    def token_delta(self, x):
        """Per-token L2 change against an empty map -- where the map 'landed'."""
        return (self.enc(x).detach() - self.empty).norm(dim=-1)[0]

    def test_different_maps_give_different_embeddings(self):
        # E3.1 guards against collapse: if the pooled vector is dominated by
        # biases, every event embeds to the same point and the head learns the
        # prior only.
        a = self.enc(blob(20, 20))
        b = self.enc(torch.randn(1, NUM_CHANNELS, H, W))
        self.assertGreater(rel_diff(a, b), 1e-3)

    def test_each_channel_changes_the_output(self):
        # E3.2 a dead channel is invisible: ch3 (pedestrian speed) could be
        # ignored entirely and the model would still train and score.
        base = self.enc(torch.zeros(1, NUM_CHANNELS, H, W))
        for c in range(NUM_CHANNELS):
            with self.subTest(channel=c):
                self.assertGreater(rel_diff(base, self.enc(blob(37, 32, channel=c))),
                                   1e-4, f'channel {c} does not affect the output')

    def test_occupancy_count_changes_the_output(self):
        # E3.3 one pedestrian vs five must not embed identically -- the least
        # demanding thing a BEV encoder should carry.
        one = self.enc(blob(30, 30, channel=1))
        many = blob(30, 30, channel=1)
        for k in range(1, 5):
            many[0, 1, 30 + 2 * k, 30] = 1.0
        self.assertGreater(rel_diff(one, self.enc(many)), 1e-3)

    def test_position_changes_the_output(self):
        # E3.4 THE spatial test. Same content, different place: a pedestrian in
        # front of the vehicle vs behind it.
        near = self.enc(blob(15, 15, channel=1))
        far = self.enc(blob(60, 50, channel=1))
        self.assertGreater(rel_diff(near, far), 1e-3,
                           'embedding is (near) translation-invariant: position is lost')

    def test_a_blob_moves_the_token_that_covers_it(self):
        # E3.5 the point of emitting tokens at all: the map is addressable. A
        # blob at grid cell (r, c) must perturb token (r // 16, c // 16) more
        # than any other. If this fails the token grid is decorative and the
        # fused branch's attention has nothing spatial to attend over.
        for r, c in [(8, 8), (24, 40), (40, 24), (56, 56), (70, 8)]:
            with self.subTest(cell=(r, c)):
                delta = self.token_delta(blob(r, c, channel=1))
                k = int(delta.argmax())
                self.assertEqual((k // TOKEN_W, k % TOKEN_W),
                                 (min(r // 16, TOKEN_H - 1), min(c // 16, TOKEN_W - 1)))

    def test_top_rows_are_not_dropped_by_the_pooling_stack(self):
        # E3.6 75 is not divisible by 16, so floor-mode pooling silently discards
        # input rows 64-74 -- the y = 15.5-21.0 m band, which holds the vehicle
        # y-max and the far pedestrian tail. They would still leak in through
        # conv padding (~13x attenuated), so the guard is on magnitude, not on
        # mere non-zero response.
        mid = float(self.token_delta(blob(40, 8, channel=1)).sum())
        top = float(self.token_delta(blob(70, 8, channel=1)).sum())
        self.assertGreater(top, 0.3 * mid,
                           f'top-of-grid response {top:.4f} vs mid-grid {mid:.4f}: '
                           'the pooling stack is truncating the last rows')

    def test_positional_embeddings_are_distinct_per_token(self):
        # E3.7 identical (or collapsed) position embeddings would leave the token
        # sequence permutation-invariant, throwing away absolute grid location --
        # which is meaningful here because the grid is global and fixed.
        pe = self.enc.pos_embed.detach()
        pairwise = torch.cdist(pe, pe)
        off_diag = pairwise[~torch.eye(len(pe), dtype=torch.bool)]
        self.assertGreater(float(off_diag.min()), 1e-3)


class TestRejectsMalformedInput(unittest.TestCase):
    """E4 -- negative cases. Each of these is a plausible caller mistake that
    must fail loudly rather than produce a wrong-but-shaped tensor."""

    def setUp(self):
        torch.manual_seed(0)
        self.enc = BEVEncoder().eval()

    def test_unflattened_5d_batch_is_rejected(self):
        # E4.1 the likeliest mistake: passing (B, T, 4, 75, 64) straight from the
        # BEV builder without folding time into the batch.
        with self.assertRaises(RuntimeError):
            self.enc(torch.randn(2, 3, NUM_CHANNELS, H, W))

    def test_missing_batch_dim_is_rejected(self):
        # E4.2 a single map as (4, 75, 64). Conv2d accepts unbatched 3D input, so
        # the guard that fires is GroupNorm's ValueError -- either way it must
        # not return a plausible (4, d_out) tensor of per-channel junk.
        with self.assertRaises((RuntimeError, ValueError)):
            self.enc(torch.randn(NUM_CHANNELS, H, W))

    def test_wrong_channel_count_is_rejected(self):
        # E4.3 e.g. dropping the speed channels but not re-instantiating.
        with self.assertRaises(RuntimeError):
            self.enc(torch.randn(2, 2, H, W))

    def test_channel_axis_swapped_with_spatial_is_rejected(self):
        # E4.4 (B, H, W, C) channels-last, the NumPy habit.
        with self.assertRaises(RuntimeError):
            self.enc(torch.randn(2, H, W, NUM_CHANNELS))

    def test_map_too_small_for_the_pooling_stack_is_rejected(self):
        # E4.5 a cropped or coarser-resolution grid must error, not emit a
        # degenerate token sequence.
        with self.assertRaises(RuntimeError):
            self.enc(torch.randn(2, NUM_CHANNELS, 8, 8))

    def test_map_that_pools_to_one_token_is_rejected_not_broadcast(self):
        # E4.6 the silent one: a map pooling to 1x1 gives (N, 1, d), which
        # broadcasts cleanly against the (20, d) position embedding and yields 20
        # identical tokens wearing 20 different positions -- a plausible-shaped
        # tensor of pure fiction. Must raise instead.
        with self.assertRaises(RuntimeError):
            self.enc(torch.randn(2, NUM_CHANNELS, 16, 16))

    def test_float64_input_is_rejected(self):
        # E4.7 build_event_bev returns float64; forgetting .float() must raise
        # rather than silently cast.
        with self.assertRaises(RuntimeError):
            self.enc(torch.zeros(2, NUM_CHANNELS, H, W, dtype=torch.float64))

    def test_nan_input_propagates_rather_than_being_swallowed(self):
        # E4.8 documents behaviour: a NaN in the map must show up in the output,
        # so a corrupt speed channel surfaces at once instead of poisoning
        # training runs later.
        x = torch.zeros(1, NUM_CHANNELS, H, W)
        x[0, 2, 10, 10] = float('nan')
        self.assertTrue(torch.isnan(self.enc(x)).any())


class TestSparseInputHardening(unittest.TestCase):
    """E6 -- the properties that make this encoder safe on a 99.8%-empty input.
    Each one pins a choice that measurement decided; a regression here is silent
    and shows up much later as an unstable fusion run."""

    def setUp(self):
        torch.manual_seed(0)
        self.enc = BEVEncoder()
        # A realistically sparse map: one vehicle, three pedestrians.
        self.x = torch.zeros(1, NUM_CHANNELS, H, W)
        self.x[0, 0, 30, 30] = 1.0
        self.x[0, 2, 30, 30] = 4.0
        for k, (r, c) in enumerate([(34, 33), (36, 31), (28, 40)]):
            self.x[0, 1, r, c] = 1.0
            self.x[0, 3, r, c] = 1.2

    def test_batch_composition_does_not_change_a_map_in_train_mode(self):
        # E6.1 the BatchNorm failure this encoder was hardened against: with
        # batch statistics over a sparse input, a map's own tokens moved by rel
        # 3.18 with batch composition. A (B, T)-flattened caller with
        # variable-length events produces exactly that, so normalisation must be
        # batch-independent.
        self.enc.train()
        alone = self.enc(self.x).detach()
        in_batch = self.enc(self.x.repeat(16, 1, 1, 1))[:1].detach()
        padded = self.enc(torch.cat([self.x, torch.zeros(15, NUM_CHANNELS, H, W)]))[:1].detach()
        self.assertLess(rel_diff(alone, in_batch), 1e-4)
        self.assertLess(rel_diff(alone, padded), 1e-4)

    def test_train_and_eval_modes_agree(self):
        # E6.2 follows from E6.1 but is what a caller actually trips over: no
        # running statistics means no train/eval divergence to debug.
        self.enc.train()
        train_out = self.enc(self.x).detach()
        self.enc.eval()
        torch.testing.assert_close(train_out, self.enc(self.x))

    def test_occupied_cells_stay_far_above_the_empty_background(self):
        # E6.3 pins bias=False. With a conv bias the empty background is a
        # non-zero constant that grows relative to the signal at every stage --
        # measured peak-to-background 2.4 at the last stage, versus 81 without
        # the bias, and the probe collapses to chance. Below ~5 the tokens are
        # mostly background.
        z = self.x
        with torch.no_grad():
            for layer in self.enc.encoder:
                z = layer(z)
        flat = z.flatten(2)
        bg = float(flat.median(dim=-1).values.mean())
        peak = float(flat.max(dim=-1).values.mean())
        self.assertGreater(peak / max(bg, 1e-6), 5.0,
                           f'peak {peak:.4f} vs background {bg:.4f}')

    def test_no_dead_channels_at_init_on_a_sparse_map(self):
        # E6.4 also pins bias=False: 7.8% of last-stage channels are dead at init
        # with the bias, so part of the width is wasted before training starts.
        z = self.x
        with torch.no_grad():
            for layer in self.enc.encoder:
                z = layer(z)
        self.assertEqual(float((z.amax(dim=(0, 2, 3)) <= 0).float().mean()), 0.0)

    def test_a_lone_occupied_cell_is_not_diluted_by_pooling(self):
        # E6.5 guards the risk average pooling carries: raw, it attenuates an
        # isolated cell 256x over four stages (1.0 -> 0.0039), which is one
        # pedestrian vanishing. GroupNorm rescales that away -- which is why
        # average pooling measures better than max here -- but that compensation
        # is the load-bearing part, so pin it: a single occupied cell must still
        # register on the same order as a four-cell cluster.
        lone = blob(30, 30, channel=1)
        cluster = lone.clone()
        for k, (r, c) in enumerate([(31, 30), (30, 31), (31, 31)]):
            cluster[0, 1, r, c] = 1.0
        d_lone = self.token_response(lone)
        d_cluster = self.token_response(cluster)
        self.assertGreater(d_lone, 0.5 * d_cluster,
                           f'lone cell {d_lone:.4f} vs 4-cell cluster {d_cluster:.4f}: '
                           'pooling is diluting isolated occupancy')

    def token_response(self, x):
        self.enc.eval()
        empty = self.enc(torch.zeros(1, NUM_CHANNELS, H, W)).detach()
        return float((self.enc(x).detach() - empty).norm(dim=-1).max())

    def test_an_empty_map_gives_a_constant_token_grid(self):
        # E6.6 semantics: with nothing in the scene every token must carry the
        # same content and differ only by its positional embedding. If empty maps
        # produced structure, the fused branch would attend to noise on the
        # timesteps where a track has no co-occurring pedestrian.
        self.enc.eval()
        tokens = self.enc(torch.zeros(1, NUM_CHANNELS, H, W)).detach()[0]
        content = tokens - self.enc.pos_embed.detach()
        self.assertLess(float((content - content[0]).abs().max()), 1e-5)


@unittest.skipUnless((PARQUET_DIR / f'{REAL_VIDEO}_interactions.parquet').exists(),
                     'interaction parquets not available')
class TestRealEventIntegration(unittest.TestCase):
    """E5 -- the encoder against a real event's maps, which synthetic tensors
    cannot exercise: true sparsity, real speed magnitudes, float64 from numpy."""

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        df = pd.read_parquet(PARQUET_DIR / f'{REAL_VIDEO}_interactions.parquet')
        group = df[(df['v_track_id'] == REAL_TRACK) & (df['roi'] == REAL_ROI)]
        _, bev = build_event_bev(group, BEVGrid())
        cls.x = torch.from_numpy(bev).float()

    def setUp(self):
        torch.manual_seed(0)
        self.enc = BEVEncoder().eval()

    def test_encodes_a_full_event_to_a_token_grid_per_timestep(self):
        # E5.1 (T, 4, 75, 64) -> (T, 20, 128) with the real T of this event.
        out = self.enc(self.x)
        self.assertEqual(out.shape, (self.x.shape[0], TOKEN_H * TOKEN_W, 128))
        self.assertTrue(torch.isfinite(out).all())

    def test_timesteps_do_not_all_embed_to_the_same_point(self):
        # E5.2 the vehicle moves across the event, so tokens must vary over time.
        # A constant sequence means the downstream temporal model gets nothing
        # from BEV.
        out = self.enc(self.x).detach()
        self.assertGreater(float(out.std(dim=0).mean()), 1e-4)

    def test_the_moving_vehicle_lights_up_more_than_one_token(self):
        # E5.3 on a real event the vehicle crosses the grid, so its occupancy
        # must reach several different tokens over the event. If every timestep
        # peaked on the same token, the token grid would not be tracking motion.
        out = self.enc(self.x).detach()
        peaks = {int(k) for k in out.norm(dim=-1).argmax(dim=-1)}
        self.assertGreater(len(peaks), 1)

    def test_real_maps_are_sparse_enough_to_be_a_fair_test(self):
        # E5.4 asserts the premise of E5.2: if maps were dense this would prove
        # little. Occupancy on real events is well under 1% of cells.
        occ = float((self.x[:, 0] > 0).float().mean() + (self.x[:, 1] > 0).float().mean())
        self.assertLess(occ, 0.01)


if __name__ == '__main__':
    unittest.main()
