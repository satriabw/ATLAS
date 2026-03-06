import gc
import torch
import argparse
import logging
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader, Subset, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from dataset.violation_dataset import load_violation_dataset
from models import CrossAttentionModel
from evaluation.evaluate_model import build_events_with_scores
from evaluation.run_evaluation import build_gt_events
from evaluation.ap_calculator import compute_map

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_val_ap(model, device, data_root: Path, val_video_ids: list) -> dict:
    """Compute APv/APn/mAP on val videos using the full evaluation pipeline."""
    parquet_dir = data_root / 'data' / 'processed' / 'interactions'
    labels_pkl  = data_root / 'data' / 'raw' / 'labels' / 'train_labels.pkl'
    nan_result  = {'APv': float('nan'), 'APn': float('nan'), 'mAP': float('nan')}

    if not labels_pkl.exists():
        logger.warning("Labels pkl not found, skipping AP computation")
        return nan_result

    predictions = build_events_with_scores(parquet_dir, val_video_ids, model, device)
    if not predictions:
        logger.warning("No predictions built for AP computation")
        return nan_result

    gt_events = build_gt_events(labels_pkl, val_video_ids, predictions)
    return compute_map(predictions, gt_events)


def train_epoch(model, dataloader, criterion, optimizer, device, scaler, accumulation_steps=4):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        vehicle_feat = batch['vehicle_feat'].to(device)
        ped_feat = batch['ped_feat'].to(device)
        labels = batch['label'].to(device)

        with autocast():
            logits = model(vehicle_feat, ped_feat)
            loss = criterion(logits, labels) / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    if len(dataloader) == 0:
        return float('nan'), float('nan')

    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            vehicle_feat = batch['vehicle_feat'].to(device)
            ped_feat = batch['ped_feat'].to(device)
            labels = batch['label'].to(device)

            with autocast():
                logits = model(vehicle_feat, ped_feat)
                loss = criterion(logits, labels)

            total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def train(args, train_dataset, val_dataset, criterion):
    batch_size = args.batch_size
    accumulation_steps = 4 if not args.overfit else 1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Sanity check: verify pedestrian features are not all zeros
    sample_batch = next(iter(train_loader))
    ped_nonzero = (sample_batch['ped_feat'] != 0).float().mean().item()
    veh_nonzero = (sample_batch['vehicle_feat'] != 0).float().mean().item()
    logger.info(f"Feature nonzero ratio — vehicle: {veh_nonzero:.3f}, ped: {ped_nonzero:.3f}")
    logger.info(f"Effective batch size: {batch_size * accumulation_steps} (batch_size={batch_size}, accumulation={accumulation_steps})")

    model = CrossAttentionModel(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    num_epochs = args.epochs
    best_val_apv = -1.0
    best_val_acc = 0.0

    checkpoint_dir = Path(__file__).parent / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # Extract unique video IDs from val dataset for AP evaluation
    if hasattr(val_dataset, 'indices'):
        val_labels = [val_dataset.dataset.labels[i] for i in val_dataset.indices]
    else:
        val_labels = val_dataset.labels
    val_video_ids = sorted(set(lbl.video_id for lbl in val_labels))
    logger.info(f"Val video IDs for AP: {val_video_ids}")

    data_root = Path(args.data_root)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, accumulation_steps)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        if val_loss == val_loss:  # not nan
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        else:
            logger.info("Val Loss: N/A (empty val set)")

        ap_result = compute_val_ap(model, device, data_root, val_video_ids)
        apv = ap_result['APv']
        apn = ap_result['APn']
        mAP = ap_result['mAP']
        if apv == apv:  # not nan
            logger.info(f"APv: {apv:.3f}, APn: {apn:.3f}, mAP: {mAP:.3f}")
        else:
            logger.info("APv/APn: N/A (no parquet files for val videos)")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / 1024**3
            reserved_gb = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"CUDA Memory: Allocated={allocated_gb:.2f}GB, Reserved={reserved_gb:.2f}GB")

        # Checkpoint on best APv; fall back to val_acc if APv is unavailable
        use_apv = apv == apv  # True if not nan
        should_save = (use_apv and apv > best_val_apv) or \
                      (not use_apv and val_acc == val_acc and val_acc > best_val_acc)
        if should_save:
            if use_apv:
                best_val_apv = apv
            else:
                best_val_acc = val_acc
            save_path = checkpoint_dir / ('overfit_model.pth' if args.overfit else 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'APv': apv,
                'APn': apn,
                'mAP': mAP,
            }, save_path)
            if use_apv:
                logger.info(f"Saved best model to {save_path.name} with APv: {apv:.3f}")
            else:
                logger.info(f"Saved best model to {save_path.name} with val acc: {val_acc:.2f}% (APv unavailable)")

    logger.info(f"Training completed. Best APv: {best_val_apv:.3f}" if best_val_apv >= 0 else "Training completed.")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train violation detection model')
    parser.add_argument('--data_root', type=str, default='/home/satria/Project/ATLAS',
                        help='ATLAS project root (contains data/)')
    parser.add_argument('--overfit', action='store_true', help='Overfit test mode (train=val, no held-out split)')
    parser.add_argument('--videos', nargs='+', type=int, default=None,
                        help='Video numbers to use in overfit mode (e.g. 1 2 3). Default: video_001 only.')
    parser.add_argument('--overfit_samples', type=int, default=0,
                        help='Max samples in overfit mode (0 = use all)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    data_root = Path(args.data_root)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.overfit:
        if args.videos:
            video_filter = [f"video_{n:03d}" for n in args.videos]
        else:
            video_filter = ['video_001']

        full_dataset = load_violation_dataset(
            data_root=data_root, label_file='train', num_frames=32, video_filter=video_filter,
        )
        logger.info(f"Total samples: {len(full_dataset)}")

        all_labels = [full_dataset.labels[i].annotation for i in range(len(full_dataset))]
        violations = all_labels.count(0)
        compliance = all_labels.count(1)
        logger.info(f"Label distribution: Violations (0)={violations}, Compliance (1)={compliance}")

        w = torch.tensor([3.5, 1.0], dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=w)
        logger.info(f"Class weights: violation={w[0]:.3f}, compliance={w[1]:.3f}")

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
        val_dataset = Subset(full_dataset, indices)
    else:
        logger.info("Loading train dataset from train_labels.pkl")
        full_dataset = load_violation_dataset(
            data_root=data_root, label_file='train', num_frames=32,
        )

        val_size = max(1, int(0.2 * len(full_dataset)))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        all_labels = [full_dataset.labels[i].annotation for i in train_dataset.indices]
        violations = all_labels.count(0)
        compliance = all_labels.count(1)
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        logger.info(f"Train label distribution: Violations (0)={violations}, Compliance (1)={compliance}")

        w = torch.tensor([3.5, 1.0], dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=w)
        logger.info(f"Class weights: violation={w[0]:.3f}, compliance={w[1]:.3f}")

    train(args, train_dataset, val_dataset, criterion)


if __name__ == '__main__':
    main()
