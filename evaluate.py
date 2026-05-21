"""
Inference script for DSAF-Net.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth \
                               --data_root /path/to/dataset/test \
                               --output_dir results/
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.dsafnet import DSAFNet
from data.dataset import FundusDataset
from utils.metrics import MetricAggregator


CVD_RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DSAF-Net")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True, help="Path to test split root")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_masks", action="store_true", help="Save predicted vessel masks")
    p.add_argument("--label_csv", default=None)
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device, output_dir, save_masks=False):
    model.eval()
    agg = MetricAggregator()
    results = []
    timings = []

    mask_dir = Path(output_dir) / "pred_masks"
    if save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        cvd_labels = batch["cvd_label"].to(device)
        filenames = batch["filename"]

        t0 = time.time()
        seg_logits, cvd_logits, _, _ = model(images)
        elapsed_ms = (time.time() - t0) * 1000

        agg.update(seg_logits, masks, cvd_logits, cvd_labels)
        timings.append(elapsed_ms)

        # Per-sample results
        preds = cvd_logits.argmax(dim=1).cpu().numpy()
        for i, fname in enumerate(filenames):
            r = {
                "filename": fname,
                "cvd_pred": CVD_RISK_LABELS[int(preds[i])],
                "cvd_true": CVD_RISK_LABELS[int(cvd_labels[i].item())],
                "inference_ms": elapsed_ms / len(filenames),
            }
            results.append(r)

            if save_masks:
                prob = torch.sigmoid(seg_logits[i, 0]).cpu().numpy()
                pred_mask = (prob >= 0.5).astype(np.uint8) * 255
                cv2.imwrite(str(mask_dir / fname), pred_mask)

    return agg.compute(), results, timings


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Dataset
    ds = FundusDataset(
        args.data_root,
        image_size=(args.image_size, args.image_size),
        augment=False,
        label_csv=args.label_csv,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    model = DSAFNet(
        in_channels=3,
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        num_heads=args.num_heads,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Inference
    metrics, results, timings = run_inference(
        model, loader, device, args.output_dir, save_masks=args.save_masks
    )

    # Report
    seg = metrics["segmentation"]
    cls = metrics["classification"]
    avg_ms = float(np.mean(timings))

    print("\n" + "=" * 55)
    print("  DSAF-Net Evaluation Results")
    print("=" * 55)
    print(f"  Vessel Segmentation")
    print(f"    Dice        : {seg['dice']:.4f}")
    print(f"    IoU         : {seg['iou']:.4f}")
    print(f"    Accuracy    : {seg['accuracy']:.4f}")
    print(f"    Sensitivity : {seg['sensitivity']:.4f}")
    print(f"    Specificity : {seg['specificity']:.4f}")
    print(f"  CVD Risk Classification")
    print(f"    Accuracy    : {cls['accuracy']*100:.2f}%")
    print(f"    Precision   : {cls['precision']*100:.2f}%")
    print(f"    Recall      : {cls['recall']*100:.2f}%")
    print(f"    F1-Score    : {cls['f1']*100:.2f}%")
    print(f"    AUC         : {cls['auc']:.4f}")
    print(f"  Inference time (avg): {avg_ms:.1f} ms/image")
    print("=" * 55)

    # Save results
    out = {
        "metrics": metrics,
        "avg_inference_ms": avg_ms,
        "per_sample": results,
    }
    with open(Path(args.output_dir) / "eval_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/eval_results.json")


if __name__ == "__main__":
    main()
