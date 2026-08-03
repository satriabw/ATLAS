import torch
import torch.nn as nn

WIDTHS = (32, 64, 128, 128)
GROUPS = 8  # divides every width above
# What the 75x64 grid reduces to after four MaxPool2d(2, ceil_mode=True).
TOKEN_H, TOKEN_W = 5, 4


class BEVEncoder(nn.Module):
    """Spatial tokens from a BEV map, for cross-attention in the fused branch.

    Input is time-flattened, (B*T, C, 75, 64) -- the caller folds time into the
    batch and unfolds the output. Four blocks take 75x64 -> 38x32 -> 19x16 ->
    10x8 -> 5x4, and the 5x4 map is emitted as 20 tokens in row-major order
    (token k covers a 16x16 cell block = 8x8 m at (k // 4, k % 4)), each carrying
    a learned positional embedding. No pooling to a single vector: the fused
    branch aggregates, so the layout the BEV exists to represent stays available.

    Every choice below is one row of scripts/ablate_bev_encoder.py, which changes
    exactly that thing and reports the cost -- rerun it rather than trusting this
    list. Numbers are side/near probe accuracy over 3 seeds against this
    configuration's 0.981/0.994, plus static measurements on real maps (99.8%
    empty). Three choices are honest ties, marked as such.

    bias=False      Decisive. A conv bias fills every empty cell with a constant,
                    and on a 99.8%-sparse map that constant becomes the signal:
                    with bias the probe collapses to chance (0.514/0.520), ROI
                    falls to 0.847 +- 0.189, last-stage peak-to-background drops
                    81 -> 2.4 and 7.8% of channels die at init.
    GroupNorm       Batch-independent, which BatchNorm is not: BN statistics over
                    a sparse input are dominated by how many maps in the batch
                    have content, so a map's own tokens moved by rel 3.18 with
                    batch composition -- and a (B, T)-flattened caller with
                    variable-length events produces exactly that. BN also costs
                    accuracy here (near 0.574). Dropping normalisation entirely
                    is worse still (0.646/0.591).
    AvgPool         Beats max pooling (0.959/0.919) despite max being the obvious
                    choice for sparse occupancy. Raw average pooling does dilute
                    an isolated cell 256x over four stages (1.0 -> 0.0039), but
                    the GroupNorm after each conv rescales that away, so the
                    dilution argument does not survive contact with the rest of
                    the stack. Test E6.5 pins the compensation.
    count_include_pad=False
                    Divides the ragged last window by the cells it actually
                    holds. TIE on the probe (0.978/0.984); kept because dividing
                    the edge window by 4 when it holds 2 would re-attenuate the
                    band ceil_mode exists to preserve.
    ceil_mode=True  TIE on the probe (0.979/0.990), because these synthetic tasks
                    sample the grid uniformly and barely stress its top edge. The
                    justification is coverage, not accuracy: 75 is not divisible
                    by 16, so floor pooling discards input rows 64-74 -- the
                    y = 15.5-21.0 m band holding the vehicle y-max (15.92 m) and
                    the far pedestrian tail, which then reach the output ~13x
                    attenuated through conv padding alone.
    ReLU            TIE with GELU (0.986/0.984). Kept as the cheaper and simpler
                    default, not because it measures better.
    4 stages        Three stages give 80 finer tokens but a smaller receptive
                    field, costing relative geometry (0.934/0.937). Four stages
                    localise a 4x12 m ROI at 0.998 anyway, so the finer grid buys
                    nothing.
    32/64/128/128   Halving every width costs near accuracy (0.951 vs 0.994) at
                    equal side accuracy -- the weakest margin here, roughly one
                    seed's spread.

    Input channels are not rescaled: on real maps the speed channels peak at
    5.6/7.8 and average 0.92/0.74 where occupied, the same order as the 0/1
    count channels, so there is no scale imbalance to correct.
    """

    def __init__(self, in_channels=4, d_out=128):
        super().__init__()
        blocks, cin = [], in_channels
        for cout in WIDTHS:
            blocks += [
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.GroupNorm(GROUPS, cout),
                nn.ReLU(),
                nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            ]
            cin = cout
        self.encoder = nn.Sequential(*blocks)
        self.num_tokens = TOKEN_H * TOKEN_W
        self.proj = nn.Linear(cin, d_out)
        self.pos_embed = nn.Parameter(torch.randn(self.num_tokens, d_out) * 0.02)
        self.output_dim = d_out

    def forward(self, x):
        # x: (B*T, C, 75, 64)
        f = self.encoder(x)  # (B*T, 128, 5, 4)
        n_tokens = f.shape[2] * f.shape[3]
        if n_tokens != self.num_tokens:
            # A differently sized map silently broadcasts against pos_embed when
            # it pools to a single cell, so reject it here.
            raise RuntimeError(
                f'expected a map that pools to {TOKEN_H}x{TOKEN_W} '
                f'({self.num_tokens} tokens), got {f.shape[2]}x{f.shape[3]}')
        tokens = f.flatten(2).transpose(1, 2)  # (B*T, 20, 128), row-major
        return self.proj(tokens) + self.pos_embed  # (B*T, 20, d_out)
