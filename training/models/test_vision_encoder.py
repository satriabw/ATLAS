"""
Run from training/:  python -m unittest models.test_vision_encoder -v

The encoder consumes time-flattened quadrant crops (B*T, 3, 224, 224) and returns
a 7x7 grid of 49 spatial tokens per frame. Most failure modes here are silent: a
trunk left in train mode couples a clip to its batch-mates, a scrambled token
order still has the right shape, a crop that reduces to 1x1 broadcasts against
pos_embed instead of raising, and a thawed trunk trains without complaint.
"""
import unittest
from pathlib import Path

import numpy as np
import torch

from .vision_encoder import VisionEncoder, TOKEN_H, TOKEN_W, TAP_CHANNELS
from dataset.vision_crop import (SIZE, crop_clip, event_frame_grid,   # noqa: F401
                                 quadrant_rect, parse_label)

ROOT = Path(__file__).resolve().parents[2]
PARQUET_DIR = ROOT / 'data/processed/interactions'
FRAME_DB = ROOT / 'data/raw/video/frames_db.h5'
REAL_VIDEO, REAL_TRACK, REAL_ROI, REAL_DOWN = 'video_001', 7, 'TOP', True

_SHARED = {}


def encoder(pretrained=False, **kw):
    """Cached so the suite builds each distinct configuration only once."""
    key = (pretrained, tuple(sorted(kw.items())))
    if key not in _SHARED:
        torch.manual_seed(0)
        _SHARED[key] = VisionEncoder(pretrained=pretrained, **kw).eval()
    return _SHARED[key]


def crops(n=2, size=SIZE, channels=3, seed=0):
    """Plausible ImageNet-normalized input: zero-mean, unit-ish scale."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, channels, size, size, generator=g)


def rel_diff(a, b):
    """L2 distance between two embeddings, scaled by their typical norm."""
    a, b = a.detach(), b.detach()
    scale = (a.norm() + b.norm()) / 2
    return float((a - b).norm() / scale.clamp(min=1e-8))


class TestOutputContract(unittest.TestCase):
    """E1 -- shape and dtype, the part a caller wires into a fusion model."""

    def test_output_shape_is_frames_by_tokens_by_d_out(self):
        out = encoder()(crops(3))
        self.assertEqual(out.shape, (3, TOKEN_H * TOKEN_W, 128))

    def test_d_out_is_honoured(self):
        self.assertEqual(encoder(d_out=64)(crops(2)).shape[-1], 64)

    def test_coarser_grid_is_honoured(self):
        enc = encoder(grid=3)
        self.assertEqual(enc.num_tokens, 9)
        self.assertEqual(enc(crops(2)).shape, (2, 9, 128))

    def test_batch_of_one_works(self):
        self.assertEqual(encoder()(crops(1)).shape, (1, 49, 128))

    def test_output_is_float32(self):
        self.assertEqual(encoder()(crops(2)).dtype, torch.float32)

    def test_eval_mode_is_deterministic(self):
        enc, x = encoder(), crops(2)
        self.assertTrue(torch.equal(enc(x), enc(x)))

    def test_matches_bev_encoder_interface(self):
        enc = encoder()
        self.assertTrue(hasattr(enc, 'num_tokens') and hasattr(enc, 'output_dim'))
        self.assertEqual(enc(crops(2)).shape, (2, enc.num_tokens, enc.output_dim))


class TestNumericalHealth(unittest.TestCase):
    """E2 -- finite values and gradients that reach exactly what they should."""

    def test_uniform_input_gives_finite_output(self):
        for fill in (0.0, -2.0, 3.0):
            self.assertTrue(torch.isfinite(
                encoder()(torch.full((2, 3, SIZE, SIZE), fill))).all())

    def test_random_input_is_finite(self):
        self.assertTrue(torch.isfinite(encoder()(crops(4))).all())

    def test_every_head_parameter_receives_gradient(self):
        enc = VisionEncoder(pretrained=False)
        enc(crops(2)).sum().backward()
        head = {'proj.weight', 'proj.bias', 'pos_embed'}
        for name, p in enc.named_parameters():
            if name in head:
                self.assertIsNotNone(p.grad, f'{name} got no gradient')
                self.assertGreater(float(p.grad.abs().sum()), 0.0, name)

    def test_trunk_receives_no_gradient(self):
        enc = VisionEncoder(pretrained=False)
        enc(crops(2)).sum().backward()
        for name, p in enc.trunk.named_parameters():
            self.assertFalse(p.requires_grad, f'trunk.{name} is trainable')
            self.assertIsNone(p.grad, f'trunk.{name} accumulated a gradient')

    def test_nan_input_propagates_rather_than_being_swallowed(self):
        x = crops(2)
        x[0, 0, 0, 0] = float('nan')
        self.assertTrue(torch.isnan(encoder()(x)[0]).any())


class TestFeatureSensitivity(unittest.TestCase):
    """E3 -- the tokens must actually encode the picture and its layout."""

    def test_different_crops_give_different_tokens(self):
        """Structured inputs, not two noise draws: independent Gaussian noise has
        near-identical statistics everywhere, so a random trunk maps both to
        nearly the same tokens (measured 0.054) without anything being wrong."""
        enc = encoder()
        stripes = torch.zeros(1, 3, SIZE, SIZE)
        stripes[:, :, ::16] = 3.0
        blob = torch.zeros(1, 3, SIZE, SIZE)
        blob[:, :, 80:140, 80:140] = 3.0
        self.assertGreater(rel_diff(enc(stripes), enc(blob)), 0.1)
        self.assertEqual(rel_diff(enc(stripes), enc(stripes)), 0.0)

    def test_each_colour_channel_changes_the_output(self):
        enc, base = encoder(), crops(1)
        for c in range(3):
            x = base.clone()
            x[:, c] += 1.0
            self.assertGreater(rel_diff(enc(base), enc(x)), 1e-3, f'channel {c} inert')

    def test_a_blanked_patch_moves_the_token_that_covers_it(self):
        """Token k must be the one that sees block k -- row-major, not transposed."""
        enc, base = encoder(pretrained=True), crops(1)
        step = SIZE // TOKEN_H
        hits = 0
        cells = [(r, c) for r in range(TOKEN_H) for c in range(TOKEN_W)]
        for r, c in cells:
            x = base.clone()
            x[:, :, r * step:(r + 1) * step, c * step:(c + 1) * step] = 0.0
            moved = (enc(base) - enc(x)).norm(dim=-1)[0].argmax().item()
            hits += int(moved == r * TOKEN_W + c)
        self.assertGreater(hits / len(cells), 0.7,
                           f'only {hits}/{len(cells)} patches moved their own token')

    def test_moving_a_patch_moves_a_different_token(self):
        enc, base = encoder(), crops(1)
        step = SIZE // TOKEN_H
        a, b = base.clone(), base.clone()
        a[:, :, 0:step, 0:step] = 0.0
        b[:, :, 0:step, 6 * step:7 * step] = 0.0
        ta = (enc(base) - enc(a)).norm(dim=-1)[0].argmax().item()
        tb = (enc(base) - enc(b)).norm(dim=-1)[0].argmax().item()
        self.assertNotEqual(ta, tb)

    def test_positional_embeddings_are_distinct_per_token(self):
        pos = encoder().pos_embed.detach()
        pair = torch.cdist(pos, pos) + torch.eye(pos.shape[0]) * 1e3
        self.assertGreater(float(pair.min()), 0.0, 'two tokens share a position')

    def test_tokens_are_not_all_identical(self):
        """A collapsed grid would still have the right shape."""
        t = encoder()(crops(1))[0].detach()
        self.assertGreater(float(t.std(dim=0).mean()), 1e-4, 'token grid is constant')


class TestRejectsMalformedInput(unittest.TestCase):
    """E4 -- shapes that would otherwise be silently reinterpreted."""

    def test_unflattened_5d_batch_is_rejected(self):
        with self.assertRaises(RuntimeError):
            encoder()(torch.randn(2, 4, 3, SIZE, SIZE))

    def test_missing_batch_dim_is_rejected(self):
        with self.assertRaises(RuntimeError):
            encoder()(torch.randn(3, SIZE, SIZE))

    def test_wrong_channel_count_is_rejected(self):
        with self.assertRaises(RuntimeError):
            encoder()(crops(2, channels=4))

    def test_channel_axis_swapped_with_spatial_is_rejected(self):
        with self.assertRaises(RuntimeError):
            encoder()(torch.randn(2, SIZE, SIZE, 3))

    def test_crop_that_reduces_to_one_token_is_rejected_not_broadcast(self):
        with self.assertRaises(RuntimeError):
            encoder()(crops(2, size=32))

    def test_float64_input_is_rejected(self):
        with self.assertRaises(RuntimeError):
            encoder()(crops(2).double())


class TestFrozenTrunkHardening(unittest.TestCase):
    """E5 -- ResNet-50 ships BatchNorm; a (B, T)-flattened caller is exactly the
    workload that turns BatchNorm into a bug. These pin the countermeasure."""

    def test_batch_composition_does_not_change_a_clip_in_train_mode(self):
        enc = encoder(pretrained=True).train()
        x = crops(8, seed=3)
        solo = enc(x[:1])
        batched = enc(x)[:1]
        self.assertLess(rel_diff(solo, batched), 1e-5,
                        'tokens depend on batch-mates -- trunk BatchNorm is live')

    def test_train_and_eval_modes_agree(self):
        enc, x = encoder(pretrained=True), crops(4, seed=4)
        self.assertLess(rel_diff(enc.train()(x), enc.eval()(x)), 1e-6)

    def test_calling_train_leaves_the_trunk_in_eval(self):
        enc = encoder().train()
        self.assertTrue(enc.training, 'the module itself should follow the caller')
        for m in enc.trunk.modules():
            self.assertFalse(m.training, f'{type(m).__name__} is in train mode')

    def test_trunk_stays_frozen_across_mode_switches(self):
        enc = encoder()
        for _ in range(2):
            enc.train(); enc.eval()
        self.assertFalse(any(p.requires_grad for p in enc.trunk.parameters()))

    def test_batchnorm_running_stats_do_not_drift(self):
        """Eval-mode BN must not update running_mean/var, or repeated forwards
        would silently change the encoder's behaviour over training."""
        enc = encoder(pretrained=True).train()
        bn = enc.trunk[1]
        before = bn.running_mean.clone()
        for _ in range(3):
            enc(crops(4, seed=7))
        self.assertTrue(torch.equal(before, bn.running_mean))

    def test_no_dead_tokens_at_init(self):
        t = encoder(pretrained=True)(crops(8, seed=5))
        per_token = t.detach().norm(dim=-1).mean(0)
        self.assertGreater(float(per_token.min()), 0.0, 'a token is always zero')


@unittest.skipUnless(FRAME_DB.exists(), 'frame database not available')
class TestRealEventIntegration(unittest.TestCase):
    """E6 -- the synthetic tensors above cannot catch a wrong crop rectangle."""

    @classmethod
    def setUpClass(cls):
        import h5py
        import pandas as pd
        df = pd.read_parquet(PARQUET_DIR / f'{REAL_VIDEO}_interactions.parquet')
        g = df[(df['v_track_id'] == REAL_TRACK) & (df['roi'] == REAL_ROI)]
        grid = event_frame_grid(g, num_frames=6)
        with h5py.File(FRAME_DB, 'r') as db:
            cls.clip = torch.from_numpy(
                crop_clip(db[REAL_VIDEO], grid, quadrant_rect(REAL_ROI, REAL_DOWN)))
        cls.enc = VisionEncoder(pretrained=True).eval()

    def test_encodes_a_real_event_to_a_token_grid_per_frame(self):
        out = self.enc(self.clip)
        self.assertEqual(out.shape, (len(self.clip), 49, 128))
        self.assertTrue(torch.isfinite(out).all())

    def test_frames_do_not_all_embed_to_the_same_point(self):
        out = self.enc(self.clip)
        self.assertGreater(rel_diff(out[0], out[-1]), 1e-3)

    def test_real_crop_is_normalized_to_roughly_unit_scale(self):
        """A crop fed without ImageNet normalization is a silent distribution shift."""
        self.assertLess(abs(float(self.clip.mean())), 2.0)
        self.assertGreater(float(self.clip.std()), 0.3)

    def test_tokens_vary_across_the_grid_on_a_real_crop(self):
        t = self.enc(self.clip)[0].detach()
        self.assertGreater(float(t.std(dim=0).mean()), 1e-3)


if __name__ == '__main__':
    unittest.main()
