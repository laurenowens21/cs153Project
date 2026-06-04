"""
filter_images.py

Reads data/Data_Entry_2017.csv and copies chest X-ray images for rare finding
classes (Pneumothorax, Emphysema) into class-specific subdirectories
under data/rare_findings/. Also copies a balanced set of "No Finding" images
to serve as the negative class.

Only pure-label rows (exactly one finding) are kept to avoid multi-label noise.

Usage:
    python scripts/filter_images.py [--csv data/Data_Entry_2017.csv]
                                    [--images-dir data]
                                    [--output-dir data/rare_findings]
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

# ── Defaults ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DEFAULT_CSV = ROOT / "data" / "Data_Entry_2017.csv"
DEFAULT_IMAGES_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "rare_findings"

RARE_FINDINGS = ["Pneumothorax", "Emphysema"]
NEGATIVE_CLASS = "No Finding"


def load_pure_label_rows(csv_path: Path, label: str) -> list[str]:
    """
    Return image filenames whose Finding Labels column is exactly `label`
    (i.e. no pipe-separated co-labels).
    """
    df = pd.read_csv(csv_path)
    mask = df["Finding Labels"] == label
    return df.loc[mask, "Image Index"].tolist()


def copy_images(
    filenames: list[str],
    source_dir: Path,
    dest_dir: Path,
    limit: int | None = None,
) -> tuple[int, int]:
    """
    Copy images from source_dir into dest_dir.

    Args:
        filenames:  List of image filenames to look for.
        source_dir: Directory (or directory tree) to search for each file.
        dest_dir:   Destination directory; created if it doesn't exist.
        limit:      If set, stop after copying this many images.

    Returns:
        (found, copied) counts. found = file existed on disk; copied = actually written.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    # NIH images may be nested in subdirectories (images_001/, images_002/, …)
    # Build a filename → path index on first call to avoid repeated glob searches.
    index = {p.name: p for p in source_dir.rglob("*.png")}
    index.update({p.name: p for p in source_dir.rglob("*.jpg")})

    found = 0
    copied = 0
    for filename in filenames:
        if limit is not None and copied >= limit:
            break
        src = index.get(filename)
        if src is None:
            continue
        found += 1
        dst = dest_dir / filename
        if not dst.exists():
            shutil.copy2(src, dst)
        copied += 1

    return found, copied


def main(csv_path: Path, images_dir: Path, output_dir: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Download Data_Entry_2017.csv from Kaggle and place it in data/."
        )
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Images directory not found: {images_dir}\n"
            "Download the NIH ChestX-ray14 images and place them in data/."
        )

    print(f"Reading {csv_path.name} …")

    # ── Collect filenames per rare class ──────────────────────────────────────
    class_filenames: dict[str, list[str]] = {}
    for finding in RARE_FINDINGS:
        filenames = load_pure_label_rows(csv_path, finding)
        class_filenames[finding] = filenames
        print(f"  {finding}: {len(filenames):>5} rows in CSV")

    # Negative class: "No Finding" rows
    no_finding_filenames = load_pure_label_rows(csv_path, NEGATIVE_CLASS)
    print(f"  {NEGATIVE_CLASS}: {len(no_finding_filenames):>5} rows in CSV")

    # ── Copy rare-class images ─────────────────────────────────────────────────
    print(f"\nCopying images to {output_dir} …\n")
    counts: dict[str, dict[str, int]] = {}
    max_rare_copied = 0

    for finding in RARE_FINDINGS:
        dest = output_dir / finding
        found, copied = copy_images(class_filenames[finding], images_dir, dest)
        counts[finding] = {"csv_rows": len(class_filenames[finding]), "found": found, "copied": copied}
        max_rare_copied = max(max_rare_copied, copied)

    # ── Copy balanced negative-class images ───────────────────────────────────
    # Match the size of the largest rare class so the negative set isn't overwhelming.
    neg_dest = output_dir / "No_Finding"
    neg_found, neg_copied = copy_images(
        no_finding_filenames, images_dir, neg_dest, limit=max_rare_copied
    )
    counts[NEGATIVE_CLASS] = {
        "csv_rows": len(no_finding_filenames),
        "found": neg_found,
        "copied": neg_copied,
        "limit": max_rare_copied,
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"{'Class':<16} {'CSV rows':>9} {'Found on disk':>14} {'Copied':>8}")
    print("─" * 52)
    for cls, c in counts.items():
        note = f"  (limit={c['limit']})" if "limit" in c else ""
        print(f"{cls:<16} {c['csv_rows']:>9} {c['found']:>14} {c['copied']:>8}{note}")

    total = sum(c["copied"] for c in counts.values())
    print(f"\nTotal images copied: {total}")
    print(f"Output directory:    {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter and copy rare-finding chest X-rays into class subdirectories."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to Data_Entry_2017.csv (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help=f"Directory containing NIH image PNGs (default: {DEFAULT_IMAGES_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root output directory for filtered images (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()
    main(csv_path=args.csv, images_dir=args.images_dir, output_dir=args.output_dir)
