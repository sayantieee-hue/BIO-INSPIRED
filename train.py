"""
Training script for DSAF-Net.

Usage:
    python scripts/train.py --data_root /path/to/dataset \
                            --epochs 100 \
                            --batch_size 4 \
                            --lr 1e-4 \
                            --save_dir checkpoints/

Dataset must follow:
    <data_root>/
      train/images/  train/masks/
      val/images/    val/masks/
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.dsafnet import DSAFNet
from models.losses import DSAFNetLoss
from data.dataset import FundusDataset
from utils.metrics import MetricAggregator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }, path)
    print(f"  ✓ Checkpoint saved: {path}")


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = {"total": 0, "seg": 0, "cls": 0, "cons": 0}
    agg = MetricAggregator()

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        cvd_labels = batch["cvd_label"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                seg_logits, cvd_logits, Fs, Fw = model(images)
                losses = criterion(seg_logits, masks, cvd_logits, cvd_labels, Fs, Fw)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            seg_logits, cvd_logits, Fs, Fw = model(images)
            losses = criterion(seg_logits, masks, cvd_logits, cvd_labels, Fs, Fw)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        for k in total_loss:
            total_loss[k] += losses[k].item()
        agg.update(seg_logits, masks, cvd_logits, cvd_labels)

    n = len(loader)
    avg_loss = {k: v / n for k, v in total_loss.items()}
    metrics = agg.compute()
    return avg_loss, metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = {"total": 0, "seg": 0, "cls": 0, "cons": 0}
    agg = MetricAggregator()

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        cvd_labels = batch["cvd_label"].to(device)

        seg_logits, cvd_logits, Fs, Fw = model(images)
        losses = criterion(seg_logits, masks, cvd_logits, cvd_labels, Fs, Fw)

        for k in total_loss:
            total_loss[k] += losses[k].item()
        agg.update(seg_logits, masks, cvd_logits, cvd_labels)

    n = len(loader)
    avg_loss = {k: v / n for k, v in total_loss.items()}
    metrics = agg.compute()
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train DSAF-Net")
    p.add_argument("--data_root", required=True, help="Dataset root (must contain train/ and val/)")
    p.add_argument("--save_dir", default="checkpoints", help="Directory to save checkpoints")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--lambda_seg", type=float, default=0.4)
    p.add_argument("--lambda_cls", type=float, default=0.4)
    p.add_argument("--lambda_cons", type=float, default=0.2)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="Use mixed precision training")
    p.add_argument("--label_csv", default=None, help="Path to pre-computed CVD label CSV")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # Directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    img_sz = (args.image_size, args.image_size)
    train_ds = FundusDataset(
        os.path.join(args.data_root, "train"),
        image_size=img_sz,
        augment=True,
        label_csv=args.label_csv,
    )
    val_ds = FundusDataset(
        os.path.join(args.data_root, "val"),
        image_size=img_sz,
        augment=False,
        label_csv=args.label_csv,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    log(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Model
    model = DSAFNet(
        in_channels=3,
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        num_heads=args.num_heads,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"Model parameters: {total_params:.2f}M")

    # Loss & optimiser
    criterion = DSAFNetLoss(args.lambda_seg, args.lambda_cls, args.lambda_cons, args.tau)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    # Training
    best_val_dice = 0.0
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        log(f"Epoch {epoch}/{args.epochs}")

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        val_dice = val_metrics["segmentation"]["dice"]
        val_cvd_acc = val_metrics["classification"]["accuracy"]

        log(
            f"  Train loss: {train_loss['total']:.4f} | "
            f"Val Dice: {val_dice:.4f} | Val CVD Acc: {val_cvd_acc*100:.2f}%"
        )

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            no_improve = 0
            save_checkpoint(
                model, optimizer, epoch,
                {"val_dice": val_dice, "val_cvd_acc": val_cvd_acc},
                save_dir / "best_model.pth",
            )
        else:
            no_improve += 1

        # Periodic save
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, {}, save_dir / f"epoch_{epoch:03d}.pth")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_cvd_acc": val_cvd_acc,
        })

        # Early stopping
        if no_improve >= args.patience:
            log(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
            break

    # Save training history
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    log(f"Training complete. Best val Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
