"""
evaluate.py

Runs a controlled data-augmentation experiment comparing four training conditions
on rare chest X-ray findings (Pneumothorax, Emphysema, No_Finding):

  Baseline  — real images only
  Prompt A  — real + synthetic images generated with the Generic prompt strategy
  Prompt B  — real + synthetic images generated with the Clinically Detailed strategy
  Prompt C  — real + synthetic images generated with the Few-shot strategy

All four conditions are evaluated on the same held-out test set (real images only,
stratified 80/20 split). Per-class and macro AUC and F1 are reported.

Outputs
-------
  results/baseline_metrics.json
  results/promptA_metrics.json
  results/promptB_metrics.json
  results/promptC_metrics.json
  results/baseline_confusion_matrix.png
  results/promptA_confusion_matrix.png
  results/promptB_confusion_matrix.png
  results/promptC_confusion_matrix.png
  results/prompt_comparison.json   — all four conditions in one file

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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import models, transforms

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
REAL_DIR = ROOT / "data" / "rare_findings"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────

CLASSES = ["Emphysema", "No_Finding", "Pneumothorax"]  # sorted for reproducibility
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

DEVICE = torch.device("cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Keys match the directory suffix used by generate_reports.py
PROMPT_STRATEGIES = {
    "promptA": "Generic",
    "promptB": "Clinically Detailed",
    "promptC": "Few-shot",
}


# ── Dataset ───────────────────────────────────────────────────────────────────


class ImageFolderFlat(Dataset):
    """
    Image dataset with a <root>/<class>/<file> layout.
    Only subdirectories present in `classes` are loaded; others are ignored.
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
        return [label for _, label in self.samples]


# ── Model ─────────────────────────────────────────────────────────────────────


def build_model(num_classes: int) -> nn.Module:
    """MobileNetV3-Small with ImageNet weights and a replaced classification head."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


# ── Training ──────────────────────────────────────────────────────────────────


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
        print(f"    Epoch {epoch:2d}/{epochs}  loss={running_loss / len(loader.dataset):.4f}")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Return (true_labels, softmax_probabilities) for the full loader."""
    model.eval()
    all_labels, all_probs = [], []
    for images, labels in loader:
        probs = torch.softmax(model(images.to(DEVICE)), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.vstack(all_probs)


# ── Metrics ───────────────────────────────────────────────────────────────────


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: list[str],
) -> dict:
    """Return per-class and macro AUC and F1 as a JSON-serialisable dict."""
    y_pred = y_prob.argmax(axis=1)
    n = len(classes)

    per_class_auc = {}
    for i, cls in enumerate(classes):
        try:
            auc = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
        except ValueError:
            auc = float("nan")
        per_class_auc[cls] = round(float(auc), 4)

    macro_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    f1_per = f1_score(y_true, y_pred, average=None, labels=list(range(n)), zero_division=0)

    return {
        "per_class_auc": per_class_auc,
        "macro_auc": round(float(macro_auc), 4),
        "per_class_f1": {cls: round(float(f1_per[i]), 4) for i, cls in enumerate(classes)},
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    output_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    labels = [c.replace("_", " ") for c in classes]
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix → {output_path.name}")


# ── Helpers ───────────────────────────────────────────────────────────────────


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    return torch.tensor(len(labels) / (num_classes * counts), dtype=torch.float32)


def stratified_split(
    dataset: ImageFolderFlat, test_size: float, seed: int
) -> tuple[Subset, Subset]:
    labels = dataset.labels()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))
    return Subset(dataset, train_idx.tolist()), Subset(dataset, test_idx.tolist())


def run_condition(
    condition_key: str,
    condition_label: str,
    train_dataset: Dataset,
    train_labels: list[int],
    test_loader: DataLoader,
    y_true: np.ndarray,
    num_classes: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict:
    """
    Train a fresh model on train_dataset, evaluate on the fixed test set,
    save metrics JSON and confusion matrix PNG, and return the metrics dict.
    """
    print(f"\n{'=' * 60}")
    print(f"  {condition_label}")
    print(f"{'=' * 60}")
    print(f"  Training samples: {len(train_dataset)}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    weights = compute_class_weights(train_labels, num_classes)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = build_model(num_classes)
    train(model, loader, epochs, weights)

    _, y_prob = predict(model, test_loader)
    y_pred = y_prob.argmax(axis=1)

    metrics = compute_metrics(y_true, y_prob, CLASSES)
    print(f"\n  Macro AUC={metrics['macro_auc']:.4f}  Macro F1={metrics['macro_f1']:.4f}")
    print(classification_report(
        y_true, y_pred,
        target_names=[c.replace("_", " ") for c in CLASSES],
        zero_division=0,
    ))

    metrics_path = RESULTS_DIR / f"{condition_key}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix(
        y_true, y_pred, CLASSES,
        RESULTS_DIR / f"{condition_key}_confusion_matrix.png",
        title=condition_label,
    )

    return metrics


# ── Comparison table ──────────────────────────────────────────────────────────


def print_comparison_table(all_results: dict[str, dict], classes: list[str]) -> None:
    """Print a multi-condition side-by-side AUC and F1 table."""
    conditions = list(all_results.keys())
    col = 11

    # Header
    print(f"\n{'─' * 80}")
    header = f"{'Class':<18}"
    for cond in conditions:
        header += f"  {'AUC-' + cond:>{col}}  {'F1-' + cond:>{col}}"
    print(header)
    print(f"{'─' * 80}")

    for cls in classes:
        row = f"{cls.replace('_', ' '):<18}"
        for cond in conditions:
            m = all_results[cond]
            row += f"  {m['per_class_auc'][cls]:>{col}.4f}  {m['per_class_f1'][cls]:>{col}.4f}"
        print(row)

    print(f"{'─' * 80}")
    macro_row = f"{'Macro':<18}"
    for cond in conditions:
        m = all_results[cond]
        macro_row += f"  {m['macro_auc']:>{col}.4f}  {m['macro_f1']:>{col}.4f}"
    print(macro_row)
    print(f"{'─' * 80}\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main(epochs: int, batch_size: int, seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_classes = len(CLASSES)

    # ── Load and split real dataset ───────────────────────────────────────────
    print("Loading real images …")
    real_ds = ImageFolderFlat(REAL_DIR, CLASSES, transform=TRANSFORM)

    if len(real_ds) == 0:
        raise RuntimeError(
            f"No images found in {REAL_DIR}. Run scripts/filter_images.py first."
        )

    per_class = {cls: 0 for cls in CLASSES}
    for _, label in real_ds.samples:
        per_class[CLASSES[label]] += 1
    for cls, n in per_class.items():
        print(f"  {cls.replace('_', ' '):<16} {n:>5} images")
    print(f"  {'Total':<16} {len(real_ds):>5} images\n")

    # Fixed train/test split — shared across all conditions
    train_real, test_ds = stratified_split(real_ds, test_size=0.2, seed=seed)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Cache y_true once; all conditions use the same test set
    y_true, _ = predict(build_model(num_classes), test_loader)
    # Rebuild test labels directly from subset (no forward pass needed)
    y_true = np.array([real_ds.samples[i][1] for i in test_ds.indices])

    train_labels_real = [real_ds.samples[i][1] for i in train_real.indices]

    all_results: dict[str, dict] = {}

    # ── Baseline ──────────────────────────────────────────────────────────────
    all_results["baseline"] = run_condition(
        condition_key="baseline",
        condition_label="Baseline — real images only",
        train_dataset=train_real,
        train_labels=train_labels_real,
        test_loader=test_loader,
        y_true=y_true,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )

    # ── Augmented conditions (one per prompt strategy) ────────────────────────
    for key, label in PROMPT_STRATEGIES.items():
        synth_dir = ROOT / "data" / f"synthetic_images_{key}"

        if not synth_dir.exists():
            print(f"\n  [SKIP] {key}: directory not found ({synth_dir.name}). "
                  "Run src/generate_reports.py first.")
            continue

        synth_ds = ImageFolderFlat(synth_dir, CLASSES, transform=TRANSFORM)

        if len(synth_ds) == 0:
            print(f"\n  [SKIP] {key}: no images found in {synth_dir.name}.")
            continue

        synth_per_class = {cls: 0 for cls in CLASSES}
        for _, lbl in synth_ds.samples:
            synth_per_class[CLASSES[lbl]] += 1
        synth_summary = ", ".join(
            f"{c.replace('_',' ')}:{n}" for c, n in synth_per_class.items() if n
        )
        print(f"\n  {key} synthetic: {synth_summary}")

        augmented_train = ConcatDataset([train_real, synth_ds])
        augmented_labels = train_labels_real + [lbl for _, lbl in synth_ds.samples]

        all_results[key] = run_condition(
            condition_key=key,
            condition_label=f"Augmented — {label} prompt",
            train_dataset=augmented_train,
            train_labels=augmented_labels,
            test_loader=test_loader,
            y_true=y_true,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )

    # ── Save combined results ─────────────────────────────────────────────────
    comparison_path = RESULTS_DIR / "prompt_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved → {comparison_path.name}")

    # ── Print comparison table ────────────────────────────────────────────────
    if len(all_results) > 1:
        print_comparison_table(all_results, CLASSES)
    else:
        print("Only one condition completed — skipping comparison table.")

    print(f"All outputs written to {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare MobileNetV3 classifiers trained with different prompt strategies."
    )
    parser.add_argument("--epochs",     type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--seed",       type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
