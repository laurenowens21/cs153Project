"""
evaluate.py

Trains a MobileNetV3-Small binary classifier on rare chest X-ray findings under
two conditions:
    (A) real images only
    (B) real images + synthetic images

Outputs per-condition AUC, F1, and confusion matrix plots to results/.

The script expects:
    data/rare_findings/<finding>/*.png       — real NIH images (from scripts/filter_images.py)
    data/synthetic_images/<finding>/*.png    — synthetic images from generate_images.py

Usage:
    python src/evaluate.py [--finding Pneumothorax] [--epochs 10] [--batch-size 32]
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REAL_IMAGE_DIR = DATA_DIR / "rare_findings"
SYNTH_IMAGE_DIR = DATA_DIR / "synthetic_images"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ───────────────────────────────────────────────────────────────────

# ImageNet normalization values — used even for grayscale (replicated to 3ch)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),  # MobileNet expects 3-ch
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class ChestXRayDataset(Dataset):
    """
    Binary dataset: label 1 for the target finding, 0 for No Finding (negative class).

    Args:
        image_paths: List of (path, label) tuples.
        transform:   torchvision transform pipeline.
    """

    def __init__(self, image_paths: list[tuple[Path, int]], transform=None):
        self.samples = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def collect_samples(
    finding: str,
    include_synthetic: bool,
    max_negatives: int = 500,
) -> list[tuple[Path, int]]:
    """
    Gather (path, label) pairs for one binary classification problem.

    Positives: real images of `finding` + (optionally) synthetic images.
    Negatives: real images labelled 'No Finding', capped at max_negatives.

    Args:
        finding:           Target rare finding class name.
        include_synthetic: Whether to include synthetic images as positives.
        max_negatives:     Cap on negative-class samples to limit class imbalance.
    """
    samples = []

    # Positive class: real images
    real_pos_dir = REAL_IMAGE_DIR / finding
    if real_pos_dir.exists():
        for p in real_pos_dir.iterdir():
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                samples.append((p, 1))

    # Positive class: synthetic images (condition B only)
    if include_synthetic:
        synth_pos_dir = SYNTH_IMAGE_DIR / finding
        if synth_pos_dir.exists():
            for p in synth_pos_dir.iterdir():
                if p.suffix.lower() == ".png" and p.name != "failed.json":
                    samples.append((p, 1))

    # Negative class: 'No Finding' real images (filter_images.py uses underscore)
    neg_dir = REAL_IMAGE_DIR / "No_Finding"
    if neg_dir.exists():
        neg_paths = [
            p for p in neg_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ][:max_negatives]
        samples.extend((p, 0) for p in neg_paths)

    if not samples:
        raise RuntimeError(
            f"No images found for finding '{finding}'. "
            "Ensure real images are in data/rare_findings/<finding>/ "
            "(run scripts/filter_images.py) and synthetic images are in "
            "data/synthetic_images/<finding>/ (run scripts/generate_reports.py "
            "then src/generate_images.py)."
        )

    return samples


def build_model() -> nn.Module:
    """Return a MobileNetV3-Small with a binary output head."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # Replace the classifier head for binary classification
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model.to(DEVICE)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Return (true_labels, predicted_probabilities) for the entire loader."""
    model.eval()
    all_labels, all_probs = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_probs)


def run_experiment(
    finding: str,
    include_synthetic: bool,
    epochs: int,
    batch_size: int,
    seed: int = 42,
) -> dict:
    """
    Full train/eval loop for one experimental condition.

    Returns a dict with AUC, F1, and per-class counts.
    """
    label = "real+synthetic" if include_synthetic else "real_only"
    print(f"\n{'='*60}")
    print(f"  Condition: {label}  |  Finding: {finding}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    samples = collect_samples(finding, include_synthetic)
    n_pos = sum(1 for _, l in samples if l == 1)
    n_neg = len(samples) - n_pos
    print(f"  Positives: {n_pos}  |  Negatives: {n_neg}  |  Total: {len(samples)}")

    # 80/20 train-test split
    n_train = int(0.8 * len(samples))
    n_test = len(samples) - n_train
    train_raw, test_raw = random_split(
        samples, [n_train, n_test], generator=torch.Generator().manual_seed(seed)
    )

    train_ds = ChestXRayDataset(list(train_raw), transform=TRAIN_TRANSFORM)
    test_ds = ChestXRayDataset(list(test_raw), transform=EVAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model()

    # Positive-class weighting to handle imbalance
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()
        if epoch % max(1, epochs // 5) == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={loss:.4f}")

    y_true, y_prob = evaluate_model(model, test_loader)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n  AUC={auc:.4f}  F1={f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=["No Finding", finding]))

    # ── Save confusion matrix plot ─────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Finding", finding]).plot(ax=ax)
    ax.set_title(f"{finding} — {label}")
    cm_path = RESULTS_DIR / f"cm_{finding}_{label}.png"
    fig.savefig(cm_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {cm_path}")

    # ── Save ROC curve ────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC — {finding} ({label})")
    ax.legend()
    roc_path = RESULTS_DIR / f"roc_{finding}_{label}.png"
    fig.savefig(roc_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return {
        "finding": finding,
        "condition": label,
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "auc": round(float(auc), 4),
        "f1": round(float(f1), 4),
    }


def main(finding: str, epochs: int, batch_size: int) -> None:
    """Run both conditions and write a summary JSON to results/."""
    results = []
    for include_synth in [False, True]:
        metrics = run_experiment(finding, include_synth, epochs, batch_size)
        results.append(metrics)

    summary_path = RESULTS_DIR / f"summary_{finding}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved → {summary_path}")

    # Side-by-side comparison
    a, b = results
    print(f"\n{'─'*40}")
    print(f"{'Metric':<10} {'Real Only':>12} {'Real+Synth':>12}")
    print(f"{'─'*40}")
    print(f"{'AUC':<10} {a['auc']:>12.4f} {b['auc']:>12.4f}")
    print(f"{'F1':<10} {a['f1']:>12.4f} {b['f1']:>12.4f}")
    print(f"{'─'*40}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate MobileNetV3 classifier.")
    parser.add_argument(
        "--finding",
        type=str,
        default="Pneumothorax",
        choices=["Pneumothorax", "Emphysema"],
        help="Rare finding class to classify (default: Pneumothorax)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    args = parser.parse_args()
    main(finding=args.finding, epochs=args.epochs, batch_size=args.batch_size)
