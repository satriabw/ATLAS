"""
Fusion Architecture Sanity Check — overfit FusedModel on video_001.

Confirms: gradient flow through the two-stage cross-attention design
(vehicle→pedestrian, then visual→trajectory) is unobstructed.

Outputs:
  results/images/overfit_fusion_loss.png
  results/analysis/overfit_fusion.json
"""
import sys, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader

TRAINING_DIR = Path(__file__).resolve().parents[1] / 'training'
sys.path.insert(0, str(TRAINING_DIR))
from dataset import load_violation_dataset
from models import FusedModel
import train as _train  # reuse _forward from train.py

DATA_ROOT    = Path('/home/satria/Project/ATLAS')
OUTPUT_DIR   = DATA_ROOT / 'results/images'
ANALYSIS_DIR = DATA_ROOT / 'results/analysis'
EPOCHS       = 50
LR           = 1e-4
BATCH_SIZE   = 8


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        logits, labels = _train._forward(model, batch, device)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_predictions(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            logits, labels = _train._forward(model, batch, device)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total   += labels.size(0)
    return correct, total


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading video_001 dataset (with vision)...")
    dataset = load_violation_dataset(
        data_root=DATA_ROOT, label_file='train',
        num_frames=32, top_k=5, video_filter='video_001', use_vision=True,
    )
    n_viol = sum(1 for l in dataset.labels if l.annotation == 0)
    n_comp = sum(1 for l in dataset.labels if l.annotation == 1)
    print(f"  {len(dataset)} samples — violations: {n_viol}, compliance: {n_comp}")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FusedModel(num_classes=2, top_k=5, num_frames=32, freeze_vision=False).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  FusedModel trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()   # equal weights — pure memorisation check

    losses = []
    for epoch in range(EPOCHS):
        loss = run_epoch(model, loader, criterion, optimizer, device)
        losses.append(loss)
        if (epoch + 1) % 25 == 0:
            print(f"  epoch {epoch+1:3d}/{EPOCHS}  loss={loss:.4f}")

    correct, total = eval_predictions(model, eval_loader, device)
    final_acc = correct / total
    print(f"\nFinal training accuracy: {correct}/{total} ({100*final_acc:.1f}%)")
    print(f"Final loss: {losses[-1]:.4f}")

    # Save JSON
    with open(ANALYSIS_DIR / 'overfit_fusion.json', 'w') as f:
        json.dump({
            'epoch_losses': losses,
            'final_loss':   losses[-1],
            'final_acc':    final_acc,
            'n_samples':    total,
            'n_epochs':     EPOCHS,
        }, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────────
    sns.set_theme(style='whitegrid', font_scale=1.0)
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs_x = np.arange(1, EPOCHS + 1)
    ax.plot(epochs_x, losses, color='#4A90D9', linewidth=2)
    ax.axhline(losses[-1], color='#2E8B57', linestyle='--', linewidth=1.2,
               label=f'Final loss = {losses[-1]:.4f}')

    # Mark first epoch where loss drops below 0.1
    conv = next((i + 1 for i, l in enumerate(losses) if l < 0.10), None)
    if conv:
        ax.axvline(conv, color='#E05C5C', linestyle=':', linewidth=1.2,
                   label=f'Loss < 0.10 at epoch {conv}')
        ax.scatter([conv], [losses[conv - 1]], color='#E05C5C', zorder=5, s=50)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax.set_title(
        'Fusion Architecture Sanity Check\n'
        f'FusedModel — video_001 only  ({total} samples)',
        fontsize=12, fontweight='bold',
    )
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = OUTPUT_DIR / 'overfit_fusion_loss.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")


if __name__ == '__main__':
    main()
