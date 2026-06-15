import random
import sys
import yaml
import numpy as np
import torch
import argparse
import logging
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import load_violation_dataset
from evaluation.ap_calculator import compute_ap
from models import CrossAttentionModel, FusedModel, VisionOnlyModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _notify(message):
    # Best-effort WhatsApp notification; notify lives under scripts/ so training has no hard dependency on it.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
        from notify import send_whatsapp
        send_whatsapp(message)
    except Exception as e:
        logger.warning(f"Notification skipped: {e}")


def _forward(model, batch, device, vision_only=False):
    labels = batch['label'].to(device)

    if vision_only:
        return model(batch['frames'].to(device)), labels

    vehicle_feat   = batch['vehicle_feat'].to(device)
    ped_feat       = batch['ped_feat'].to(device)
    v_padding_mask = batch['v_padding_mask'].to(device)
    p_padding_mask = batch['p_padding_mask'].to(device)

    if 'frames' in batch:
        logits = model(vehicle_feat, ped_feat, batch['frames'].to(device), v_padding_mask, p_padding_mask)
        return logits, labels

    logits = model(vehicle_feat, ped_feat, v_padding_mask, p_padding_mask)
    return logits, labels


def _color_jitter(frames):
    # Light brightness/contrast jitter on the RGB channels (train only).
    # Applied in normalized space; mask channels (3:) untouched.
    B = frames.shape[0]
    a = torch.empty(B, 1, 1, 1, 1, device=frames.device).uniform_(0.85, 1.15)
    b = torch.empty(B, 1, 1, 1, 1, device=frames.device).uniform_(-0.15, 0.15)
    frames = frames.clone()
    frames[:, :, :3] = frames[:, :, :3] * a + b
    return frames


def train_epoch(model, dataloader, criterion, optimizer, device, scaler, amp_enabled, vision_only=False, clip_norm=5.0):
    model.train()
    total_loss, correct, total, total_norm = 0, 0, 0, 0.0

    for batch in tqdm(dataloader, desc="Training"):
        if 'frames' in batch:
            batch['frames'] = _color_jitter(batch['frames'])
        with torch.autocast(device_type='cuda', enabled=amp_enabled):
            logits, labels = _forward(model, batch, device, vision_only)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        total_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm).item()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)

    n = len(dataloader)
    return total_loss / n, 100 * correct / total, total_norm / n


def validate(model, dataloader, criterion, device, vision_only=False):
    if len(dataloader) == 0:
        return float('nan'), float('nan'), float('nan')

    model.eval()
    total_loss, correct, total = 0, 0, 0
    preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            logits, labels = _forward(model, batch, device, vision_only)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)
            p_viol = torch.softmax(logits.float(), dim=1)[:, 0]
            preds.extend({'gt_label': int(y), 'score': float(s)}
                         for y, s in zip(labels.cpu(), p_viol.cpu()))

    val_apv = compute_ap(preds, target_class=0, score_key='score')
    return total_loss / len(dataloader), 100 * correct / total, val_apv


def _scene_split(full_dataset, seed):
    all_videos = sorted({lbl.video_id for lbl in full_dataset.labels})
    random.seed(seed)
    random.shuffle(all_videos)
    n_val       = max(1, round(0.15 * len(all_videos)))
    val_videos  = set(all_videos[-n_val:])
    train_videos = set(all_videos) - val_videos

    train_idx = [i for i, lbl in enumerate(full_dataset.labels) if lbl.video_id in train_videos]
    val_idx   = [i for i, lbl in enumerate(full_dataset.labels) if lbl.video_id in val_videos]

    logger.info(f"Train videos ({len(train_videos)}): {sorted(train_videos)}")
    logger.info(f"Val videos   ({len(val_videos)}): {sorted(val_videos)}")
    return Subset(full_dataset, train_idx), Subset(full_dataset, val_idx), train_idx


def train(args, train_dataset, val_dataset, criterion):
    import wandb

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    if args.vision_only:
        model = VisionOnlyModel(num_classes=2, num_frames=32, backbone=args.backbone,
                                freeze_vision=args.freeze).to(device)
    elif args.fused:
        model = FusedModel(num_classes=2, top_k=args.top_k, num_frames=32,
                           freeze_vision=args.freeze).to(device)
    else:
        model = CrossAttentionModel(num_classes=2, top_k=args.top_k, num_frames=32).to(device)

    # Pretrained ResNet layers fine-tune at a lower LR than the fresh heads/GRUs.
    # The inflated conv1 (features.0) carries fresh mask-channel weights, so it
    # trains at the full LR with the heads.
    def _is_backbone(n):
        return n.startswith('vision_encoder.features') and not n.startswith('vision_encoder.features.0.')
    backbone = [p for n, p in model.named_parameters() if p.requires_grad and _is_backbone(n)]
    rest     = [p for n, p in model.named_parameters() if p.requires_grad and not _is_backbone(n)]
    param_groups = [{'params': rest}] + ([{'params': backbone, 'lr': args.backbone_lr}] if backbone else [])
    optimizer   = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    amp_enabled = not args.no_amp and device.type == 'cuda'
    scaler      = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    logger.info(f"AMP: {'enabled' if amp_enabled else 'disabled'}")

    checkpoint_dir = Path(__file__).parent / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    type_tag   = 'vision' if args.vision_only else 'fused' if args.fused else 'traj'
    model_type = {'vision': 'vision', 'fused': 'fused', 'traj': 'cross_attention'}[type_tag]
    ckpt_stem  = {'vision': 'vision', 'fused': 'fused', 'traj': 'model'}[type_tag]

    run_name = args.run_name or f"{type_tag}_seed{args.seed}"

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    if not args.no_notify:
        _notify(f"ATLAS run '{run_name}' started — {args.epochs} epochs, model={type_tag}")

    overfit_prefix = 'overfit_' if args.overfit else ''
    best_ckpt_name = f'best_{overfit_prefix}{ckpt_stem}.pth'
    best_val_apv   = float('-inf')
    patience_count = 0

    def _save_ckpt(path, extra=None):
        d = {
            'epoch': epoch, 'model_type': model_type,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc, 'val_loss': val_loss, 'val_apv': val_apv,
            'backbone': getattr(args, 'backbone', 'resnet18'),
            'h5': getattr(args, 'h5', 'frames.h5'),
        }
        if extra:
            d.update(extra)
        torch.save(d, path)

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc, grad_norm = train_epoch(model, train_loader, criterion, optimizer, device, scaler, amp_enabled, args.vision_only, clip_norm=args.clip_norm)
        val_loss,   val_acc,   val_apv   = validate(model, val_loader, criterion, device, args.vision_only)
        lr = optimizer.param_groups[0]['lr']

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Grad Norm: {grad_norm:.4f}")
        if val_loss:
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val APv: {val_apv:.4f}")
            scheduler.step(val_loss)
        else:
            logger.info("Val Loss: N/A (empty val set)")

        if not args.no_wandb:
            wandb.log({
                'train/loss': train_loss, 'train/acc': train_acc,
                'val/loss':   val_loss,   'val/acc':   val_acc,
                'val/apv':    val_apv,
                'lr':         lr,         'grad_norm': grad_norm,
                'epoch':      epoch + 1,
            })

        # best checkpoint — selected on val APv (ranking metric, matches the
        # evaluation); val loss diverges under miscalibration while ranking
        # still improves, so loss-based selection picked epoch-1 models.
        if val_apv == val_apv and val_apv > best_val_apv:
            best_val_apv   = val_apv
            patience_count = 0
            _save_ckpt(checkpoint_dir / best_ckpt_name)
            logger.info(f"Saved best model → {best_ckpt_name} (val_apv: {val_apv:.4f})")
        else:
            patience_count += 1

        # periodic checkpoint
        if (epoch + 1) % 5 == 0:
            _save_ckpt(checkpoint_dir / f'epoch_{epoch+1:03d}_{type_tag}.pth')

        if patience_count >= args.patience:
            logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break

    if not args.no_wandb:
        wandb.finish()

    if not args.no_notify:
        _notify(f"ATLAS run '{run_name}' done — best val APv: {best_val_apv:.4f}")
    logger.info(f"Training completed. Best val APv: {best_val_apv:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train violation detection model')
    parser.add_argument('--config',         type=str,   default=None)
    parser.add_argument('--data_root',      type=str,   default='/home/satria/Project/ATLAS')
    parser.add_argument('--videos',         nargs='+',  type=int, default=None)
    parser.add_argument('--epochs',         type=int,   default=20)
    parser.add_argument('--batch_size',     type=int,   default=2)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--backbone_lr',    type=float, default=1e-5)
    parser.add_argument('--weight_decay',   type=float, default=1e-4)
    parser.add_argument('--clip_norm',      type=float, default=5.0)
    parser.add_argument('--top_k',          type=int,   default=5)
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--patience',       type=int,   default=15)
    parser.add_argument('--wandb_project',  type=str,   default='ATLAS')
    parser.add_argument('--run_name',       type=str,   default=None)
    parser.add_argument('--no_wandb',       action='store_true')
    parser.add_argument('--no_notify',      action='store_true')
    parser.add_argument('--no_amp',         action='store_true')
    parser.add_argument('--mode',           choices=['trajectory', 'vision', 'fused'], default='trajectory')
    parser.add_argument('--backbone',       choices=['resnet18', 'r2plus1d'], default='resnet18')
    parser.add_argument('--freeze',         choices=['early', 'full'], default='early',
                        help="'full' freezes the whole vision backbone (head-only probe); inflated conv1 stays trainable")
    parser.add_argument('--h5',             type=str, default='frames.h5',
                        help="frames h5 file name under data/raw/video/ (e.g. frames_r2.h5)")
    parser.add_argument('--overfit',        action='store_true')

    # load YAML config first so CLI args override it
    pre = parser.parse_known_args()[0]
    if pre.config:
        with open(pre.config) as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**{k: v for k, v in cfg.items() if v is not None})

    args = parser.parse_args()
    args.vision_only = args.mode == 'vision'
    args.fused       = args.mode == 'fused'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_root = Path(args.data_root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # both fused and vision-only need video frames loaded
    use_vision = args.fused or args.vision_only

    if args.overfit:
        logger.info("=== OVERFIT MODE: video_001 only ===")
        full_dataset  = load_violation_dataset(
            data_root=data_root, label_file='train', num_frames=32, top_k=args.top_k,
            video_filter='video_001', use_vision=use_vision, h5_name=args.h5,
        )
        train_dataset = full_dataset
        val_dataset   = full_dataset
        train_labels  = [lbl.annotation for lbl in full_dataset.labels]
    else:
        logger.info("Loading train dataset from train_labels.pkl")
        full_dataset  = load_violation_dataset(
            data_root=data_root, label_file='train', num_frames=32, top_k=args.top_k,
            video_filter=args.videos, use_vision=use_vision, h5_name=args.h5,
        )
        train_dataset, val_dataset, train_idx = _scene_split(full_dataset, args.seed)
        train_labels = [full_dataset.labels[i].annotation for i in train_idx]

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Train label distribution: Violations (0)={train_labels.count(0)}, Compliance (1)={train_labels.count(1)}")

    if args.vision_only:
        # Class weights push a weak vision branch toward a constant prior;
        # selection on val APv handles the imbalance instead.
        criterion = torch.nn.CrossEntropyLoss()
    else:
        w = torch.tensor([3.5, 1.0], dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=w)

    train(args, train_dataset, val_dataset, criterion)


if __name__ == '__main__':
    main()
