import torch
import torch.nn as nn
import torchvision

TAP_CHANNELS = 1024          # ResNet-50 layer3 width
NATIVE_H, NATIVE_W = 14, 14  # what a 224x224 crop reduces to at layer3
TOKEN_H, TOKEN_W = 7, 7      # what we pool it to


class VisionEncoder(nn.Module):
    """Spatial tokens from a quadrant RGB crop, for cross-attention in the fused branch.

    Input is time-flattened, (B*T, 3, 224, 224) -- the caller folds time into the
    batch and unfolds the output, exactly as BEVEncoder does, so the two branches
    present the same interface to fusion. A 224x224 crop reduces to 14x14 at
    layer3, average-pooled to 7x7 and emitted as 49 tokens in row-major order
    (token k covers a 32x32 px block of the resized crop at (k // 7, k % 7)),
    each carrying a learned positional embedding. No pooling to a single vector:
    the fused branch aggregates, so the layout the crop exists to represent
    stays available.

    Input must be ImageNet-normalized (dataset.vision_crop.crop_clip does this);
    the trunk was fitted under those statistics.

    The trunk is inherited, not derived -- ResNet-50's internal layers are not
    ours to justify. What IS justified below is everything we chose around it.
    Every choice is one row of scripts/ablate_vision_encoder.py, which changes
    exactly that thing and reports the cost -- rerun it rather than trusting this
    list. APv is the mean over 3 seeds on that script's subsampled bed (1000
    events/split, 4 frames, violation prevalence 0.245), where this configuration
    scores 0.4305 +- 0.0036. Those absolutes sit below the full-bed ledger
    numbers by construction; rows are comparable to each other, not to the
    ledger. Three choices are honest ties, marked as such.

    tap=layer3      Decisive, and the choice that reverses an earlier conclusion.
                    Tapping layer4 instead costs 0.3688 (-0.062) and its tokens
                    are barely local: blanking a patch moves the token covering
                    it only 71% of the time, against 100% at layer3. layer4's
                    receptive field is wide enough that a 7x7 cell no longer
                    corresponds to its own patch, and the readout stops
                    mattering there (layer4 flat 0.3688 vs layer4 gap 0.3857 --
                    pooling is actually AHEAD). At layer3 the readout is worth
                    +0.098. The 2026-08-03 full-bed probe that found flat > gap
                    at layer4 (+0.0355) does NOT replicate on this bed, where
                    gap leads by 0.017; layer3 is where that effect is real and
                    large rather than marginal and bed-dependent.
    flat readout    Decisive. Mean-pooling the 49 tokens costs 0.3324 (-0.098)
                    and attention-pooling 0.3333 (-0.097) -- a learned attention
                    pool is worth nothing over a plain mean, the same ordering
                    BEVEncoder's probe found. Pooling all the way to one token
                    (global pool) also lands at 0.3324, i.e. the whole cost is
                    paid the moment the grid collapses, however it collapses.
    7x7 pooling     TIE with layer3's native 14x14 grid (0.4302 vs 0.4305) at a
                    quarter of the tokens, so the finer grid buys nothing and
                    49 tokens keeps cross-attention cheap. Coarser does cost:
                    5x5 0.3991 (-0.031), 3x3 0.3834 (-0.047), 1x1 0.3324.
    BatchNorm eval  TIE on the probe (train mode 0.4298), kept for stability
                    rather than accuracy. ResNet-50 ships BatchNorm; in train
                    mode a clip's features move by relative 0.545 when its
                    batch-mates change, against 0.000 in eval. A (B, T)-flattened
                    caller reshuffles batch composition every step, so train-mode
                    BatchNorm makes a clip's tokens a function of what it happened
                    to be batched with. This is the vision analogue of the
                    BEVEncoder GroupNorm row.
    ImageNet init   Decisive. A randomly initialised trunk scores 0.3187 (-0.112),
                    leaves 0.9% of channels dead, and drops locality to 0.83. The
                    prior is load-bearing, not a convenience.
    d_out=128       TIE across 64 (0.4262), 128 (0.4305) and 256 (0.4335, sd
                    0.0098 -- inside its own seed spread). 128 is chosen to match
                    BEVEncoder.output_dim so fusion sees one token width.
    pos_embed       TIE (off scores 0.4301). Kept for a structural reason the
                    probe cannot see: the flat readout reads tokens in a fixed
                    order, so it never needs them to identify themselves, but
                    cross-attention does. Remove it only alongside a readout that
                    demonstrably does not need token identity.

    Trunk frozen throughout. Fine-tuning it is a fusion-time decision, not an
    encoder one, and this repo has a leakage history with fine-tuned vision
    features (2026-07-09 gated-r2: fine-tuned feats scored 0.726 against 0.81
    frozen, traced to train-set memorization) -- so the default is the
    leakage-free one and any unfreezing has to be argued at the point it is done.
    """

    def __init__(self, d_out=128, grid=TOKEN_H, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet50(weights=weights)
        # Stop at layer3: layer4's receptive field destroys token locality.
        self.trunk = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                   net.layer1, net.layer2, net.layer3)
        for p in self.trunk.parameters():
            p.requires_grad = False
        self.pool = nn.Identity() if grid == NATIVE_H else nn.AdaptiveAvgPool2d(grid)
        self.grid = grid
        self.num_tokens = grid * grid
        self.proj = nn.Linear(TAP_CHANNELS, d_out)
        self.pos_embed = nn.Parameter(torch.randn(self.num_tokens, d_out) * 0.02)
        self.output_dim = d_out
        self.trunk.eval()

    def train(self, mode=True):
        """Keep the frozen trunk in eval mode regardless of the parent's mode.

        Without this, the trunk's BatchNorm normalizes by batch statistics and a
        clip's tokens become a function of its batch-mates -- see the docstring's
        BatchNorm row for the measured coupling.
        """
        super().train(mode)
        self.trunk.eval()
        return self

    def forward(self, x):
        # x: (B*T, 3, 224, 224), ImageNet-normalized
        if x.dim() != 4 or x.shape[1] != 3:
            raise RuntimeError(
                f'expected time-flattened (B*T, 3, H, W), got {tuple(x.shape)}')
        with torch.no_grad():
            f = self.trunk(x)                   # (B*T, 1024, 14, 14)
        if min(f.shape[2], f.shape[3]) < self.grid:
            # AdaptiveAvgPool2d would happily UPSAMPLE a smaller map by
            # replication, fabricating tokens that share the same pixels, so a
            # too-small crop has to be rejected here rather than pooled.
            raise RuntimeError(
                f'crop reduces to {f.shape[2]}x{f.shape[3]} at the tap, smaller '
                f'than the requested {self.grid}x{self.grid} token grid')
        f = self.pool(f)
        tokens = f.flatten(2).transpose(1, 2)   # (B*T, 49, 1024), row-major
        return self.proj(tokens) + self.pos_embed
