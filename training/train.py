import gc
import random
import torch
import argparse
import logging
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from dataset.violation_dataset import load_violation_dataset, SpeedStats
from models import CrossAttentionModel
from evaluation.evaluate_model import build_events_with_scores
from evaluation.ap_calculator import compute_map

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_val_ap(
    model, device, data_root: Path, val_video_ids: list,
    speed_stats: SpeedStats = None,
) -> dict:
    """Compute APv/APn/mAP on val videos using the full evaluation pipeline."""
    parquet_dir = data_root / 'data' / 'processed' / 'interactions'
    labels_pkl  = data_root / 'data' / 'raw' / 'labels' / 'train_labels.pkl'
    nan_result  = {'APv': float('nan'), 'APn': float('nan'), 'mAP': float('nan')}

    if not labels_pkl.exists():
        logger.warning("Labels pkl not found, skipping AP computation")
        return nan_result

    predictions = build_events_with_scores(
        parquet_dir, labels_pkl, val_video_ids, model, device,
        speed_stats=speed_stats,
    )
    if not predictions:
        logger.warning("No predictions built for AP computation")
        return nan_result

    return compute_map(predictions)


def _forward(model, batch, device):
    """Move batch to device and run forward pass, returning (logits, labels)."""
    vehicle_feat   = batch['vehicle_feat'].to(device)
    ped_feat       = batch['ped_feat'].to(device)
    v_padding_mask = batch['v_padding_mask'].to(device)
    p_padding_mask = batch['p_padding_mask'].to(device)
    labels         = batch['label'].to(device)
    logits = model(vehicle_feat, ped_feat, v_padding_mask, p_padding_mask)
    return logits, labels


def train_epoch(model, dataloader, criterion, optimizer, device, scaler, accumulation_steps=4):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        with autocast():
            logits, labels = _forward(model, batch, device)
            loss = criterion(logits, labels) / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)

    return total_loss / len(dataloader), 100 * correct / total


def validate(model, dataloader, criterion, device):
    if len(dataloader) == 0:
        return float('nan'), float('nan')

    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            with autocast():
                logits, labels = _forward(model, batch, device)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

    return total_loss / len(dataloader), 100 * correct / total


def train(args, train_dataset, val_dataset, criterion, speed_stats: SpeedStats = None):
    batch_size         = args.batch_size
    accumulation_steps = 4 if not args.overfit else 1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    sample_batch = next(iter(train_loader))
    ped_nonzero = (sample_batch['ped_feat'] != 0).float().mean().item()
    veh_nonzero = (sample_batch['vehicle_feat'] != 0).float().mean().item()
    logger.info(f"Feature nonzero ratio — vehicle: {veh_nonzero:.3f}, ped: {ped_nonzero:.3f}")
    logger.info(f"Effective batch size: {batch_size * accumulation_steps}")

    model     = CrossAttentionModel(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler    = GradScaler()

    best_val_apv = -1.0
    best_val_acc = 0.0

    checkpoint_dir = Path(__file__).parent / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    if hasattr(val_dataset, 'indices'):
        val_labels = [val_dataset.dataset.labels[i] for i in val_dataset.indices]
    else:
        val_labels = val_dataset.labels
    val_video_ids = sorted(set(lbl.video_id for lbl in val_labels))
    logger.info(f"Val video IDs for AP: {val_video_ids}")
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    data_root = Path(args.data_root)

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, accumulation_steps)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        if val_loss == val_loss:
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        else:
            logger.info("Val Loss: N/A (empty val set)")

        ap_result = compute_val_ap(model, device, data_root, val_video_ids, speed_stats)
        apv, apn, mAP = ap_result['APv'], ap_result['APn'], ap_result['mAP']
        if apv == apv:
            logger.info(f"APv: {apv:.3f}, APn: {apn:.3f}, mAP: {mAP:.3f}")
        else:
            logger.info("APv/APn: N/A (no parquet files for val videos)")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / 1024**3
            reserved_gb  = torch.cuda.memory_reserved()  / 1024**3
            logger.info(f"CUDA Memory: Allocated={allocated_gb:.2f}GB, Reserved={reserved_gb:.2f}GB")

        use_apv    = apv == apv
        should_save = (use_apv and apv > best_val_apv) or \
                      (not use_apv and val_acc == val_acc and val_acc > best_val_acc)
        if should_save:
            best_val_apv = apv if use_apv else best_val_apv
            best_val_acc = val_acc if not use_apv else best_val_acc
            save_path = checkpoint_dir / ('overfit_model.pth' if args.overfit else 'best_model.pth')
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':              val_acc,
                'val_loss':             val_loss,
                'APv':                  apv,
                'APn':                  apn,
                'mAP':                  mAP,
                'speed_stats':          vars(speed_stats) if speed_stats else None,
            }, save_path)
            metric_str = f"APv: {apv:.3f}" if use_apv else f"val acc: {val_acc:.2f}%"
            logger.info(f"Saved best model to {save_path.name} with {metric_str}")

    logger.info(f"Training completed. Best APv: {best_val_apv:.3f}" if best_val_apv >= 0 else "Training completed.")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train violation detection model')
    parser.add_argument('--data_root',       type=str,   default='/home/satria/Project/ATLAS')
    parser.add_argument('--overfit',         action='store_true')
    parser.add_argument('--videos',          nargs='+',  type=int,   default=None)
    parser.add_argument('--overfit_samples', type=int,   default=0)
    parser.add_argument('--epochs',          type=int,   default=20)
    parser.add_argument('--batch_size',      type=int,   default=2)
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--top_k',           type=int,   default=5,
                        help='Number of closest pedestrians to use per event')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.overfit:
        video_filter = [f"video_{n:03d}" for n in args.videos] if args.videos else ['video_001']

        full_dataset = load_violation_dataset(
            data_root=data_root, label_file='train', num_frames=32,
            top_k=args.top_k, video_filter=video_filter,
        )
        full_dataset.compute_and_set_speed_stats()  # all videos (train == val, no leakage risk)
        speed_stats = full_dataset.speed_stats

        all_labels = [full_dataset.labels[i].annotation for i in range(len(full_dataset))]
        logger.info(f"Total samples: {len(full_dataset)} — "
                    f"Violations: {all_labels.count(0)}, Compliance: {all_labels.count(1)}")

        logger.info("=" * 60)
        logger.info("OVERFIT TEST MODE")
        logger.info(f"Videos: {video_filter}")
        logger.info("Goal: model should reach ~100% accuracy to confirm it can learn")
        logger.info("=" * 60)

        if args.overfit_samples > 0:
            buckets = defaultdict(list)
            for i in range(len(full_dataset)):
                buckets[full_dataset.labels[i].annotation].append(i)
            n = min(len(buckets[0]), len(buckets[1]), args.overfit_samples // 2)
            indices = buckets[0][:n] + buckets[1][:n]
        else:
            indices = list(range(len(full_dataset)))

        selected = [full_dataset.labels[i].annotation for i in indices]
        logger.info(f"Using {len(indices)} samples — Violations: {selected.count(0)}, Compliance: {selected.count(1)}")

        train_dataset = Subset(full_dataset, indices)
        val_dataset   = Subset(full_dataset, indices)

    else:
        logger.info("Loading train dataset from train_labels.pkl")
        full_dataset = load_violation_dataset(
            data_root=data_root, label_file='train', num_frames=32, top_k=args.top_k,
        )

        # Scene-stratified 85/15 split: all events from the same video go to the same split.
        # Shuffle before splitting so the val set isn't biased toward late-recorded videos.
        all_videos = sorted({lbl.video_id for lbl in full_dataset.labels})
        random.seed(42)
        random.shuffle(all_videos)
        n_val_videos   = max(1, round(0.15 * len(all_videos)))
        val_video_set  = set(all_videos[-n_val_videos:])
        train_video_set = set(all_videos) - val_video_set

        train_indices = [i for i, lbl in enumerate(full_dataset.labels) if lbl.video_id in train_video_set]
        val_indices   = [i for i, lbl in enumerate(full_dataset.labels) if lbl.video_id in val_video_set]

        logger.info(f"Train videos ({len(train_video_set)}): {sorted(train_video_set)}")
        logger.info(f"Val videos   ({len(val_video_set)}): {sorted(val_video_set)}")

        # Compute speed stats from training videos only to avoid val leakage
        full_dataset.compute_and_set_speed_stats(train_video_set)
        speed_stats = full_dataset.speed_stats

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset   = Subset(full_dataset, val_indices)

        train_labels = [full_dataset.labels[i].annotation for i in train_indices]
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        logger.info(f"Train label distribution: Violations (0)={train_labels.count(0)}, "
                    f"Compliance (1)={train_labels.count(1)}")

    w = torch.tensor([3.5, 1.0], dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=w)
    logger.info(f"Class weights: violation={w[0]:.3f}, compliance={w[1]:.3f}")

    train(args, train_dataset, val_dataset, criterion, speed_stats)


if __name__ == '__main__':
    main()
