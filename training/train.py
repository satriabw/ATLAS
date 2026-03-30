import random
import torch
import argparse
import logging
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset.violation_dataset import load_violation_dataset
from models import CrossAttentionModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _forward(model, batch, device):
    """Move batch to device and run forward pass, returning (logits, labels)."""
    vehicle_feat   = batch['vehicle_feat'].to(device)
    ped_feat       = batch['ped_feat'].to(device)
    v_padding_mask = batch['v_padding_mask'].to(device)
    p_padding_mask = batch['p_padding_mask'].to(device)
    labels         = batch['label'].to(device)
    logits = model(vehicle_feat, ped_feat, v_padding_mask, p_padding_mask)
    return logits, labels


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in tqdm(dataloader, desc="Training"):
        logits, labels = _forward(model, batch, device)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
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
            logits, labels = _forward(model, batch, device)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

    return total_loss / len(dataloader), 100 * correct / total


def train(args, train_dataset, val_dataset, criterion):
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    model     = CrossAttentionModel(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10,
    )

    best_val_loss = float('inf')

    checkpoint_dir = Path(__file__).parent / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        if val_loss == val_loss:
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            scheduler.step(val_loss)
        else:
            logger.info("Val Loss: N/A (empty val set)")

        if val_loss == val_loss and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':              val_acc,
                'val_loss':             val_loss,
            }, save_path)
            logger.info(f"Saved best model to {save_path.name} with val_loss: {val_loss:.4f}")

    logger.info(f"Training completed. Best val loss: {best_val_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train violation detection model')
    parser.add_argument('--data_root',  type=str,   default='/home/satria/Project/ATLAS')
    parser.add_argument('--videos',     nargs='+',  type=int,   default=None)
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch_size', type=int,   default=2)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--top_k',      type=int,   default=5,
                        help='Number of closest pedestrians to use per event')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)

    train_labels = [full_dataset.labels[i].annotation for i in train_indices]
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Train label distribution: Violations (0)={train_labels.count(0)}, "
                f"Compliance (1)={train_labels.count(1)}")

    w = torch.tensor([3.5, 1.0], dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=w)
    logger.info(f"Class weights: violation={w[0]:.3f}, compliance={w[1]:.3f}")

    train(args, train_dataset, val_dataset, criterion)


if __name__ == '__main__':
    main()
