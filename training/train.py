import gc
import torch
import argparse
import logging
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from dataset.violation_dataset import load_violation_dataset
from models import TrajectoryOnlyModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, criterion, optimizer, device, scaler, accumulation_steps=4):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        trajectory = batch['trajectory'].to(device)
        labels = batch['label'].to(device)

        with autocast():
            logits = model(trajectory)
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
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            trajectory = batch['trajectory'].to(device)
            labels = batch['label'].to(device)

            with autocast():
                logits = model(trajectory)
                loss = criterion(logits, labels)

            total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def create_train_val_split(dataset, val_ratio=0.2, seed=42):
    import random
    random.seed(seed)

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    split_idx = int(len(indices) * (1 - val_ratio))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    return train_indices, val_indices


def train(args, train_dataset, val_dataset, criterion):
    batch_size = args.batch_size
    accumulation_steps = 4 if not args.overfit else 1

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Effective batch size: {batch_size * accumulation_steps} (batch_size={batch_size}, accumulation={accumulation_steps})")

    model = TrajectoryOnlyModel(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    num_epochs = args.epochs
    best_val_acc = 0

    checkpoint_dir = Path(__file__).parent / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, accumulation_steps)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / 1024**3
            reserved_gb = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"CUDA Memory: Allocated={allocated_gb:.2f}GB, Reserved={reserved_gb:.2f}GB")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = checkpoint_dir / ('overfit_model.pth' if args.overfit else 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, save_path)
            logger.info(f"Saved best model to {save_path.name} with val acc: {val_acc:.2f}%")

    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train violation detection model')
    parser.add_argument('--data_root', type=str, default='/home/satriabw/ATLAS',
                        help='ATLAS project root (contains data/)')
    parser.add_argument('--overfit', action='store_true', help='Overfit test on video_001 only')
    parser.add_argument('--overfit_samples', type=int, default=8,
                        help='Max samples for overfit test (0 = use all video_001 labels)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    data_root = Path(args.data_root)

    # Training and validation always use train_labels.
    # Overfit mode restricts to video_001 only.
    video_filter = 'video_001' if args.overfit else None

    logger.info(f"Loading train dataset from {data_root}" + (f" (video_001 only)" if video_filter else ""))
    full_dataset = load_violation_dataset(
        data_root=data_root,
        label_file='train',
        num_frames=32,
        video_filter=video_filter,
    )
    logger.info(f"Total samples: {len(full_dataset)}")

    all_labels = [full_dataset.labels[i].annotation for i in range(len(full_dataset))]
    violations = all_labels.count(0)
    compliance = all_labels.count(1)
    logger.info(f"Label distribution: Violations (0)={violations}, Compliance (1)={compliance}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weights = torch.tensor([1.0 / max(violations, 1), 1.0 / max(compliance, 1)],
                                  dtype=torch.float32, device=device)
    class_weights = class_weights / class_weights.sum()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    if args.overfit:
        logger.info("=" * 60)
        logger.info("OVERFIT TEST MODE")
        logger.info("Samples: video_001 only")
        logger.info("Goal: model should reach ~100% accuracy to confirm it can learn")
        logger.info("=" * 60)

        n = args.overfit_samples if args.overfit_samples > 0 else len(full_dataset)
        indices = list(range(min(n, len(full_dataset))))

        selected = [full_dataset.labels[i].annotation for i in indices]
        logger.info(f"Using {len(indices)} samples — Violations: {selected.count(0)}, Compliance: {selected.count(1)}")

        train_dataset = Subset(full_dataset, indices)
        val_dataset = Subset(full_dataset, indices)
    else:
        train_indices, val_indices = create_train_val_split(full_dataset, val_ratio=0.2, seed=42)
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

    train(args, train_dataset, val_dataset, criterion)


if __name__ == '__main__':
    main()
