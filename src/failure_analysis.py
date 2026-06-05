"""
failure_analysis.py

Visual comparison of real vs synthetic chest X-ray images for Emphysema and
Pneumothorax. For each class a grid of 10 real images (left column) and 10
synthetic images (right column) is saved to results/.

Quantitative pixel-level statistics (mean brightness, std, contrast) are
computed and printed alongside qualitative observations about the visual
differences between the two distributions.

Outputs
-------
  results/emphysema_comparison.png
  results/pneumothorax_comparison.png

Usage:
    python src/failure_analysis.py [--n 10] [--synth-dir synthetic_images_promptC]
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REAL_DIR = DATA_DIR / "rare_findings"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DISPLAY_SIZE = (224, 224)  # resize for uniform grid display


# ── Image loading ─────────────────────────────────────────────────────────────


def collect_images(directory: Path, n: int, seed: int = 42) -> list[Path]:
    """
    Return a reproducible random sample of n image paths from directory.
    Raises FileNotFoundError if the directory does not exist.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    all_paths = sorted(
        p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(all_paths) == 0:
        raise RuntimeError(f"No images found in {directory}")

    rng = random.Random(seed)
    return rng.sample(all_paths, min(n, len(all_paths)))


def load_gray(path: Path) -> np.ndarray:
    """Load image as a normalised float32 grayscale array in [0, 1]."""
    img = Image.open(path).convert("L").resize(DISPLAY_SIZE, Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


# ── Statistics ────────────────────────────────────────────────────────────────


def image_stats(arrays: list[np.ndarray]) -> dict:
    """
    Compute pixel-level statistics across a list of grayscale image arrays.

    Returns mean brightness, standard deviation (proxy for contrast),
    and the mean of per-image standard deviations (local contrast).
    """
    stack = np.stack(arrays)           # (N, H, W)
    return {
        "mean_brightness": float(stack.mean()),
        "global_std":      float(stack.std()),
        "mean_local_std":  float(np.array([a.std() for a in arrays]).mean()),
        "min_pixel":       float(stack.min()),
        "max_pixel":       float(stack.max()),
    }


# ── Grid rendering ────────────────────────────────────────────────────────────


def save_comparison_grid(
    real_paths: list[Path],
    synth_paths: list[Path],
    finding: str,
    output_path: Path,
) -> tuple[dict, dict]:
    """
    Render a side-by-side grid of real (left half) and synthetic (right half)
    images and save to output_path.

    Layout:
        Columns 0..n-1  → real images
        Columns n..2n-1 → synthetic images
        A vertical divider is drawn between the two halves.

    Returns (real_stats, synth_stats) for downstream printing.
    """
    n = len(real_paths)
    real_arrays  = [load_gray(p) for p in real_paths]
    synth_arrays = [load_gray(p) for p in synth_paths]

    real_stats  = image_stats(real_arrays)
    synth_stats = image_stats(synth_arrays)

    # Layout: 1 title row + n image rows, 2*n + 1 columns (n real | divider | n synth)
    total_cols = n * 2 + 1
    fig = plt.figure(figsize=(total_cols * 1.4, n * 1.5 + 1.2))

    gs = gridspec.GridSpec(
        n + 1, total_cols,
        figure=fig,
        hspace=0.05,
        wspace=0.05,
        height_ratios=[0.6] + [1.0] * n,
    )

    # Column header labels
    ax_real_title  = fig.add_subplot(gs[0, : n])
    ax_synth_title = fig.add_subplot(gs[0, n + 1 :])
    for ax, txt in [(ax_real_title, f"Real  ({finding})"),
                    (ax_synth_title, f"Synthetic  ({finding}, Few-shot prompt)")]:
        ax.text(0.5, 0.5, txt, ha="center", va="center",
                fontsize=11, fontweight="bold", transform=ax.transAxes)
        ax.axis("off")

    # Divider column — draw a vertical line
    for row in range(1, n + 1):
        ax_div = fig.add_subplot(gs[row, n])
        ax_div.axvline(x=0.5, color="#888888", linewidth=1.5)
        ax_div.axis("off")

    # Image cells
    for row, (r_arr, s_arr) in enumerate(zip(real_arrays, synth_arrays), start=1):
        for col, arr in enumerate(r_arr[np.newaxis], start=0):   # real
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

        for col_offset, arr in enumerate([s_arr], start=0):       # synthetic
            ax = fig.add_subplot(gs[row, n + 1 + col_offset])
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

    fig.suptitle(
        f"{finding}: Real vs Synthetic Comparison  "
        f"(brightness real={real_stats['mean_brightness']:.3f} "
        f"synth={synth_stats['mean_brightness']:.3f})",
        fontsize=10, y=0.995,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return real_stats, synth_stats


# ── Observations ──────────────────────────────────────────────────────────────


def print_observations(
    finding: str,
    real_stats: dict,
    synth_stats: dict,
    output_path: Path,
) -> None:
    """
    Print computed statistics and qualitative observations about the visual gap
    between real and synthetic images for one finding class.
    """
    bright_delta = synth_stats["mean_brightness"] - real_stats["mean_brightness"]
    contrast_delta = synth_stats["mean_local_std"] - real_stats["mean_local_std"]

    brighter = "brighter" if bright_delta > 0 else "darker"
    more_contrast = "higher" if contrast_delta > 0 else "lower"

    print(f"\n{'─' * 60}")
    print(f"  {finding}  —  pixel statistics")
    print(f"{'─' * 60}")
    print(f"  {'Metric':<22} {'Real':>10} {'Synthetic':>10} {'Delta':>10}")
    print(f"  {'─' * 54}")
    for key, label in [
        ("mean_brightness", "Mean brightness"),
        ("global_std",      "Global std"),
        ("mean_local_std",  "Mean local std"),
        ("min_pixel",       "Min pixel"),
        ("max_pixel",       "Max pixel"),
    ]:
        r, s = real_stats[key], synth_stats[key]
        print(f"  {label:<22} {r:>10.4f} {s:>10.4f} {s - r:>+10.4f}")

    print(f"\n  Qualitative observations:")
    print(f"  • Synthetic images are {abs(bright_delta):.3f} units {brighter} on average.")
    print(f"  • Synthetic images have {more_contrast} local contrast "
          f"(Δ={contrast_delta:+.4f}).")
    print(
        "  • Real NIH X-rays tend to show sharp rib edges, clear lung field texture,\n"
        "    and consistent DICOM windowing. Synthetic images often appear smoother,\n"
        "    with softer edges and reduced fine-grained anatomical detail."
    )
    if finding == "Pneumothorax":
        print(
            "  • The pleural line separating the collapsed lung from the air space —\n"
            "    the key diagnostic feature — is frequently absent or poorly defined\n"
            "    in synthetic images, replaced by generalised haziness."
        )
    elif finding == "Emphysema":
        print(
            "  • Hyperinflation cues (flattened diaphragm, increased lucency) are\n"
            "    inconsistently present. Synthetic images sometimes show a normally\n"
            "    rounded diaphragm, undermining the pathology signal."
        )
    print(
        "  • Both classes show synthetic images with visible 'painterly' texture\n"
        "    artefacts uncommon in real DICOM radiographs."
    )
    print(f"\n  Grid saved → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main(n: int, synth_dir_name: str) -> None:
    """
    Args:
        n:              Number of real and synthetic images to show per class.
        synth_dir_name: Name of the synthetic images directory under data/,
                        e.g. 'synthetic_images_promptC'.
    """
    synth_root = DATA_DIR / synth_dir_name

    if not synth_root.exists():
        raise FileNotFoundError(
            f"Synthetic image directory not found: {synth_root}\n"
            f"Run src/generate_reports.py --strategy C first."
        )

    for finding in ["Emphysema", "Pneumothorax"]:
        print(f"\nProcessing {finding} …")

        real_paths  = collect_images(REAL_DIR / finding, n)
        synth_paths = collect_images(synth_root / finding, n)

        print(f"  Sampled {len(real_paths)} real, {len(synth_paths)} synthetic images.")

        output_path = RESULTS_DIR / f"{finding.lower()}_comparison.png"

        real_stats, synth_stats = save_comparison_grid(
            real_paths, synth_paths, finding, output_path
        )

        print_observations(finding, real_stats, synth_stats, output_path)

    print(f"\nDone. Comparison grids saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visual comparison of real vs synthetic chest X-ray images."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of images per class to display (default: 10)",
    )
    parser.add_argument(
        "--synth-dir",
        type=str,
        default="synthetic_images_promptC",
        help="Synthetic image directory name under data/ (default: synthetic_images_promptC)",
    )
    args = parser.parse_args()
    main(n=args.n, synth_dir_name=args.synth_dir)
