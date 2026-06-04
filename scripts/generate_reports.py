"""
generate_reports.py

Generates synthetic radiology reports for Pneumothorax and Emphysema images
already filtered into data/rare_findings/. One report is produced per image
file using the Anthropic API (Claude). Results are saved to
data/synthetic_reports.json.

This script reads image filenames directly from data/rare_findings/ so it does
not require Data_Entry_2017.csv to be present.

Usage:
    python scripts/generate_reports.py [--max-per-class N]
"""

import argparse
import json
import os
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
RARE_FINDINGS_DIR = ROOT / "data" / "rare_findings"
OUTPUT_PATH = ROOT / "data" / "synthetic_reports.json"

FINDINGS = ["Pneumothorax", "Emphysema"]

SYSTEM_PROMPT = (
    "You are a board-certified radiologist writing official chest X-ray radiology reports. "
    "Your reports are factual, use standard radiological terminology, and follow the style "
    "used in academic medical centers. Each report contains four sections in order: "
    "CLINICAL INDICATION, TECHNIQUE, FINDINGS, and IMPRESSION. "
    "Do not include any patient identifiers. Write one complete report per request."
)

# Detailed clinical context fed into each per-finding prompt to produce varied,
# realistic output rather than repetitive boilerplate.
FINDING_CONTEXT = {
    "Pneumothorax": (
        "pneumothorax visible on chest X-ray. Vary the presentation realistically: "
        "it may be small and apical, moderate, large, or tension pneumothorax with "
        "mediastinal shift. The affected side (left or right) should vary across reports. "
        "Include realistic incidental findings such as mild atelectasis, pleural thickening, "
        "or rib fractures where clinically plausible."
    ),
    "Emphysema": (
        "pulmonary emphysema visible on chest X-ray. Vary the severity: mild with subtle "
        "hyperinflation, moderate with flattened diaphragms and increased AP diameter, "
        "or severe with bullae and markedly attenuated vascularity. Include realistic "
        "co-findings such as mild cardiomegaly, flattened hemidiaphragms, or signs of "
        "chronic obstructive pulmonary disease (COPD)."
    ),
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def collect_image_paths(findings_dir: Path, findings: list[str], max_per_class: int) -> list[dict]:
    """
    Walk data/rare_findings/<finding>/ for each finding and return a list of
    records with the image filename and its associated finding label.

    Args:
        findings_dir:  Root directory containing per-class subdirectories.
        findings:      List of finding class names to include.
        max_per_class: Maximum images to process per class.

    Returns:
        List of dicts with keys: image_index, finding.
    """
    records = []
    for finding in findings:
        class_dir = findings_dir / finding
        if not class_dir.exists():
            print(f"  [WARN] Directory not found, skipping: {class_dir}")
            continue

        images = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )[:max_per_class]

        for img_path in images:
            records.append({"image_index": img_path.name, "finding": finding})

        print(f"  {finding}: {len(images)} images found")

    return records


def build_user_prompt(finding: str, sample_index: int) -> str:
    """Construct the per-image user message sent to Claude."""
    context = FINDING_CONTEXT[finding]
    return (
        f"Generate a realistic chest X-ray radiology report for a patient with {context} "
        f"Sample index: {sample_index}. "
        "Return only the report text with no preamble, title, or commentary."
    )


def generate_report(client: anthropic.Anthropic, finding: str, sample_index: int) -> str:
    """Call Claude to generate one synthetic radiology report and return the text."""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": build_user_prompt(finding, sample_index)}
        ],
    )
    return response.content[0].text.strip()


def main(max_per_class: int = 50) -> None:
    """
    Main entry point.

    Iterates over filtered images in data/rare_findings/, calls Claude for each,
    and writes all reports to data/synthetic_reports.json. Already-generated
    entries are loaded first so the script is safely resumable.

    Args:
        max_per_class: Maximum number of images to generate reports for per class.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example → .env and add your key."
        )

    if not RARE_FINDINGS_DIR.exists():
        raise FileNotFoundError(
            f"Expected filtered images at {RARE_FINDINGS_DIR}. "
            "Run scripts/filter_images.py first."
        )

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Scanning {RARE_FINDINGS_DIR} …")
    records = collect_image_paths(RARE_FINDINGS_DIR, FINDINGS, max_per_class)

    if not records:
        raise RuntimeError("No images found. Check that data/rare_findings/ has subdirectories.")

    # Load any previously generated reports so the run is resumable
    existing: dict[str, dict] = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for entry in json.load(f):
                existing[entry["image_index"]] = entry
        print(f"Resuming: {len(existing)} reports already generated.\n")
    else:
        print()

    results = list(existing.values())
    to_generate = [r for r in records if r["image_index"] not in existing]
    print(f"Generating {len(to_generate)} new reports …\n")

    for record in tqdm(to_generate, desc="Generating reports"):
        finding = record["finding"]
        image_name = record["image_index"]

        try:
            report_text = generate_report(client, finding, len(results))
        except anthropic.RateLimitError:
            tqdm.write("  Rate limited — waiting 60 s …")
            time.sleep(60)
            report_text = generate_report(client, finding, len(results))

        results.append(
            {
                "image_index": image_name,
                "finding": finding,
                "report": report_text,
            }
        )

        # Save after every entry so progress survives a crash or interruption
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

        time.sleep(0.4)  # stay within API rate limits

    # ── Summary ───────────────────────────────────────────────────────────────
    counts = {f: sum(1 for r in results if r["finding"] == f) for f in FINDINGS}
    print(f"\nDone. Reports saved → {OUTPUT_PATH}")
    print(f"{'Finding':<16} {'Reports':>8}")
    print("─" * 26)
    for finding, n in counts.items():
        print(f"{finding:<16} {n:>8}")
    print(f"{'Total':<16} {sum(counts.values()):>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic radiology reports for Pneumothorax and Emphysema."
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=50,
        help="Max images to generate reports for per finding class (default: 50)",
    )
    args = parser.parse_args()
    main(max_per_class=args.max_per_class)
