"""
evaluate.py

Runs a controlled data-augmentation experiment on rare chest X-ray findings.

Condition A — Baseline:
    Train MobileNetV3-Small (ImageNet pretrained) on real images only.

Condition B — Augmented:
    Train a fresh MobileNetV3-Small (ImageNet pretrained) on real + synthetic images.

Both conditions are evaluated on the same held-out test set (real images only).
Per-class and macro AUC, F1, and confusion matrices are saved to results/.

Usage:
    python src/evaluate.py [--epochs 10] [--batch-size 32] [--seed 42]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
REAL_DIR = ROOT / "data" / "rare_findings"
SYNTH_DIR = ROOT / "data" / "synthetic_images"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────

CLASSES = ["Emphysema", "No_Finding", "Pneumothorax"]  # sorted for reproducibility
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

SEED = 42
DEVICE = torch.device("cpu")

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # X-rays are grayscale; MobileNet needs 3ch
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


# ── Dataset ───────────────────────────────────────────────────────────────────


class ImageFolderFlat(Dataset):
    """
    Minimal image dataset that reads from a root directory containing one
    subdirectory per class. Unlike torchvision's ImageFolder it accepts an
    explicit class list and ignores any extra subdirectories (e.g. Hernia).

    Args:
        root:      Directory with <class>/<image> layout.
        classes:   Ordered list of class names to include.
        transform: Transform applied to each PIL image.
    """

    def __init__(self, root: Path, classes: list[str], transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for cls in classes:
            cls_dir = root / cls
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((p, CLASS_TO_IDX[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def labels(self) -> list[int]:
        """Return all labels in sample order — used for stratified splitting."""
        return [label for _, label in self.samples]


# ── Model ─────────────────────────────────────────────────────────────────────


def build_model(num_classes: int) -> nn.Module:
    """
    Return a MobileNetV3-Small with ImageNet pretrained weights and a
    replaced classification head for num_classes outputs.
    """
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


# ── Training & Evaluation ─────────────────────────────────────────────────────


def train(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    class_weights: torch.Tensor,
) -> None:
    """Fine-tune model in place for a fixed number of epochs."""
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        scheduler.step()
        avg_loss = running_loss / len(loader.dataset)
        print(f"    Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference and return (true_labels, predicted_probabilities).
    Probabilities have shape (N, num_classes) after softmax.
    """
    model.eval()
    all_labels, all_probs = [], []
    for images, labels in loader:
        logits = model(images.to(DEVICE))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.vstack(all_probs)


# ── Metrics ───────────────────────────────────────────────────────────────────


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: list[str],
) -> dict:
    """
    Compute per-class and macro AUC and F1.

    Returns a dict ready for JSON serialisation.
    """
    y_pred = y_prob.argmax(axis=1)

    # Per-class AUC (one-vs-rest)
    per_class_auc = {}
    for i, cls in enumerate(classes):
        try:
            auc = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
        except ValueError:
            auc = float("nan")
        per_class_auc[cls] = round(float(auc), 4)

    macro_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")

    # Per-class F1
    f1_per = f1_score(y_true, y_pred, average=None, labels=list(range(len(classes))), zero_division=0)
    per_class_f1 = {cls: round(float(f1_per[i]), 4) for i, cls in enumerate(classes)}
    macro_f1 = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)

    return {
        "per_class_auc": per_class_auc,
        "macro_auc": round(float(macro_auc), 4),
        "per_class_f1": per_class_f1,
        "macro_f1": macro_f1,
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    output_path: Path,
    title: str,
) -> None:
    """Render and save a labelled confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    display_labels = [c.replace("_", " ") for c in classes]

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=display_labels).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {output_path.name}")


# ── Helpers ───────────────────────────────────────────────────────────────────


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """
    Return inverse-frequency class weights to handle imbalance.
    Weight for class c = total_samples / (num_classes * count_c).
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)  # avoid division by zero
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def stratified_split(
    dataset: ImageFolderFlat, test_size: float, seed: int
) -> tuple[Subset, Subset]:
    """Return (train_subset, test_subset) with a stratified 80/20 split."""
    labels = dataset.labels()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))
    return Subset(dataset, train_idx.tolist()), Subset(dataset, test_idx.tolist())


def print_comparison_table(
    baseline: dict, augmented: dict, classes: list[str]
) -> None:
    """Print a side-by-side AUC and F1 comparison between the two conditions."""
    col = 14
    print(f"\n{'─' * 70}")
    print(f"{'':20} {'── AUC ──':^25}    {'── F1 ──':^25}")
    print(f"{'Class':<20} {'Baseline':>{col}} {'Augmented':>{col}}    {'Baseline':>{col}} {'Augmented':>{col}}")
    print(f"{'─' * 70}")

    for cls in classes:
        b_auc = baseline["per_class_auc"][cls]
        a_auc = augmented["per_class_auc"][cls]
        b_f1 = baseline["per_class_f1"][cls]
        a_f1 = augmented["per_class_f1"][cls]
        label = cls.replace("_", " ")
        print(
            f"{label:<20} {b_auc:>{col}.4f} {a_auc:>{col}.4f}"
            f"    {b_f1:>{col}.4f} {a_f1:>{col}.4f}"
        )

    print(f"{'─' * 70}")
    print(
        f"{'Macro':20} {baseline['macro_auc']:>{col}.4f} {augmented['macro_auc']:>{col}.4f}"
        f"    {baseline['macro_f1']:>{col}.4f} {augmented['macro_f1']:>{col}.4f}"
    )
    print(f"{'─' * 70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main(epochs: int, batch_size: int, seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_classes = len(CLASSES)

    # ── Load real dataset and split ───────────────────────────────────────────
    print("Loading real images …")
    real_ds = ImageFolderFlat(REAL_DIR, CLASSES, transform=TRANSFORM)

    if len(real_ds) == 0:
        raise RuntimeError(
            f"No images found in {REAL_DIR}. "
            "Run scripts/filter_images.py first."
        )

    per_class = {cls: 0 for cls in CLASSES}
    for _, label in real_ds.samples:
        per_class[CLASSES[label]] += 1
    for cls, n in per_class.items():
        print(f"  {cls.replace('_', ' '):<16} {n:>5} images")
    print(f"  {'Total':<16} {len(real_ds):>5} images\n")

    train_real, test_ds = stratified_split(real_ds, test_size=0.2, seed=seed)

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # ── Condition A: Baseline (real only) ─────────────────────────────────────
    print("=" * 60)
    print("  Condition A: Baseline — real images only")
    print("=" * 60)

    train_labels_a = [real_ds.samples[i][1] for i in train_real.indices]
    weights_a = compute_class_weights(train_labels_a, num_classes)

    train_loader_a = DataLoader(
        train_real, batch_size=batch_size, shuffle=True, num_workers=0
    )

    model_a = build_model(num_classes)
    print(f"  Training on {len(train_real)} samples …")
    train(model_a, train_loader_a, epochs, weights_a)

    y_true, y_prob_a = predict(model_a, test_loader)
    y_pred_a = y_prob_a.argmax(axis=1)

    metrics_a = compute_metrics(y_true, y_prob_a, CLASSES)
    print(f"\n  Macro AUC={metrics_a['macro_auc']:.4f}  Macro F1={metrics_a['macro_f1']:.4f}")
    print(classification_report(
        y_true, y_pred_a,
        target_names=[c.replace("_", " ") for c in CLASSES],
        zero_division=0,
    ))

    with open(RESULTS_DIR / "baseline_metrics.json", "w") as f:
        json.dump(metrics_a, f, indent=2)

    save_confusion_matrix(
        y_true, y_pred_a, CLASSES,
        RESULTS_DIR / "baseline_confusion_matrix.png",
        title="Baseline (real only)",
    )

    # ── Condition B: Augmented (real + synthetic) ──────────────────────────────
    print("\n" + "=" * 60)
    print("  Condition B: Augmented — real + synthetic images")
    print("=" * 60)

    synth_ds = ImageFolderFlat(SYNTH_DIR, CLASSES, transform=TRANSFORM)

    if len(synth_ds) == 0:
        print(
            "  [WARN] No synthetic images found in data/synthetic_images/. "
            "Run scripts/generate_images.py to produce them.\n"
            "  Skipping augmented condition."
        )
        return

    synth_per_class = {cls: 0 for cls in CLASSES}
    for _, label in synth_ds.samples:
        synth_per_class[CLASSES[label]] += 1
    for cls, n in synth_per_class.items():
        if n:
            print(f"  {cls.replace('_', ' '):<16} {n:>5} synthetic images")
    print(f"  {'Total':<16} {len(synth_ds):>5} synthetic images\n")

    # Combine real training split + all synthetic images into one dataset
    from torch.utils.data import ConcatDataset

    augmented_train = ConcatDataset([train_real, synth_ds])

    # Recompute class weights over the combined training set
    train_labels_b = (
        train_labels_a
        + [label for _, label in synth_ds.samples]
    )
    weights_b = compute_class_weights(train_labels_b, num_classes)

    train_loader_b = DataLoader(
        augmented_train, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # Fresh model — same pretrained init, not fine-tuned condition-A weights
    model_b = build_model(num_classes)
    print(f"  Training on {len(augmented_train)} samples …")
    train(model_b, train_loader_b, epochs, weights_b)

    _, y_prob_b = predict(model_b, test_loader)
    y_pred_b = y_prob_b.argmax(axis=1)

    metrics_b = compute_metrics(y_true, y_prob_b, CLASSES)
    print(f"\n  Macro AUC={metrics_b['macro_auc']:.4f}  Macro F1={metrics_b['macro_f1']:.4f}")
    print(classification_report(
        y_true, y_pred_b,
        target_names=[c.replace("_", " ") for c in CLASSES],
        zero_division=0,
    ))

    with open(RESULTS_DIR / "augmented_metrics.json", "w") as f:
        json.dump(metrics_b, f, indent=2)

    save_confusion_matrix(
        y_true, y_pred_b, CLASSES,
        RESULTS_DIR / "augmented_confusion_matrix.png",
        title="Augmented (real + synthetic)",
    )

    # ── Comparison table ───────────────────────────────────────────────────────
    print_comparison_table(metrics_a, metrics_b, CLASSES)
    print("All results saved to results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate MobileNetV3 with and without synthetic augmentation."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
