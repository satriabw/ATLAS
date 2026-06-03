import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from dataset.trajectory import build_group_trajectory, resample_trajectory, padding_mask, DEFAULT_TOP_K
from evaluation.inference import _build_ped_stack
from models import CrossAttentionModel

DATA_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single interaction event")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pth"))
    parser.add_argument("--video",      type=str, required=True, help="e.g. video_042 or 42")
    parser.add_argument("--track",      type=int, required=True, help="v_track_id")
    parser.add_argument("--roi",        type=str, required=True, choices=["TOP", "BOT"])
    parser.add_argument("--top-k",      type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--data-root",  type=Path, default=DATA_ROOT)
    args = parser.parse_args()

    video_id = (args.video if args.video.startswith("video_")
                else f"video_{int(args.video):03d}")

    parquet_path = (args.data_root / "data" / "processed" / "interactions"
                    / f"{video_id}_interactions.parquet")
    if not parquet_path.exists():
        print(f"Parquet not found: {parquet_path}")
        sys.exit(1)

    df    = pd.read_parquet(parquet_path)
    group = df[(df["v_track_id"] == args.track) & (df["roi"] == args.roi)]
    if group.empty:
        print(f"No interaction found: video={video_id} track={args.track} roi={args.roi}")
        sys.exit(1)

    _, _, vehicle_feat_raw, ped_feats_raw = build_group_trajectory(group, args.top_k)
    v_arr, v_len = resample_trajectory(vehicle_feat_raw, args.num_frames)
    p_arr, p_mask = _build_ped_stack(ped_feats_raw, args.num_frames, args.top_k)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CrossAttentionModel(num_classes=2, top_k=args.top_k, num_frames=args.num_frames).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    v_t  = torch.from_numpy(v_arr).unsqueeze(0).to(device)
    p_t  = torch.from_numpy(p_arr).unsqueeze(0).to(device)
    vm_t = torch.from_numpy(padding_mask(v_len, args.num_frames)).unsqueeze(0).to(device)
    pm_t = torch.from_numpy(p_mask).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(v_t, p_t, vm_t, pm_t)
        probs  = F.softmax(logits, dim=1)[0]

    p_viol = probs[0].item()
    p_comp = probs[1].item()
    pred   = "VIOLATION" if p_viol >= 0.5 else "COMPLIANCE"

    print()
    print("=" * 42)
    print(f"  Event : {video_id}  track={args.track}  roi={args.roi}")
    print(f"  Result: {pred}")
    print(f"  P(violation)  = {p_viol:.4f}")
    print(f"  P(compliance) = {p_comp:.4f}")
    print("=" * 42)


if __name__ == "__main__":
    main()
