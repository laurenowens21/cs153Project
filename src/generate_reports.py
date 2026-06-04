"""
generate_reports.py

Reads NIH ChestX-ray14's Data_Entry_2017.csv, filters for rare finding classes
(Pneumothorax, Emphysema), and calls the Anthropic API to generate one
realistic synthetic radiology report per sample. Results are saved to
data/synthetic_reports.json.

Usage:
    python src/generate_reports.py [--max-per-class N]
"""

import argparse
import json
import os
import time
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
CSV_PATH = DATA_DIR / "Data_Entry_2017.csv"
OUTPUT_PATH = DATA_DIR / "synthetic_reports.json"

RARE_FINDINGS = ["Pneumothorax", "Emphysema"]

# System prompt that establishes the clinical persona for report generation.
SYSTEM_PROMPT = (
    "You are a board-certified radiologist writing official chest X-ray radiology reports. "
    "Your reports are factual, use standard radiological terminology, and follow the SOAP/BIRADS "
    "style used in academic medical centers. Each report has the sections: "
    "CLINICAL INDICATION, TECHNIQUE, FINDINGS, and IMPRESSION. "
    "Do not include any patient identifiers. Write one complete report per request."
)

FINDING_CONTEXT = {
    "Pneumothorax": (
        "pneumothorax on chest X-ray, ranging from small apical to large tension pneumothorax "
        "with mediastinal shift"
    ),
    "Emphysema": (
        "pulmonary emphysema on chest X-ray, with hyperinflation, flattened diaphragms, "
        "and increased AP diameter"
    ),
}


def build_user_prompt(finding: str, image_index: int) -> str:
    """Build the per-sample user message sent to Claude."""
    context = FINDING_CONTEXT[finding]
    return (
        f"Generate a realistic chest X-ray radiology report for a patient with {context}. "
        f"Vary the severity, side (if applicable), and incidental findings realistically. "
        f"Sample index: {image_index}. "
        "Return only the report text, with no preamble or commentary."
    )


def generate_report(client: anthropic.Anthropic, finding: str, image_index: int) -> str:
    """Call Claude to generate one synthetic radiology report."""
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": build_user_prompt(finding, image_index)}
        ],
    )
    return message.content[0].text.strip()


def load_rare_samples(csv_path: Path, findings: list[str], max_per_class: int) -> pd.DataFrame:
    """
    Load Data_Entry_2017.csv and return rows where Finding Labels contains
    exactly one of the target rare findings (pure-label rows only).
    """
    df = pd.read_csv(csv_path)

    # NIH CSV stores pipe-separated labels in 'Finding Labels'
    frames = []
    for finding in findings:
        # Keep only rows whose label set is exactly this finding (avoids multi-label noise)
        mask = df["Finding Labels"] == finding
        subset = df[mask].head(max_per_class).copy()
        subset["target_finding"] = finding
        frames.append(subset)

    return pd.concat(frames, ignore_index=True)


def main(max_per_class: int = 50) -> None:
    """
    Main entry point. Generates one synthetic report per sample and saves to JSON.

    Args:
        max_per_class: Maximum number of real samples to generate reports for,
                       per finding class. Kept low by default to respect API rate limits.
    """
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Expected NIH metadata CSV at {CSV_PATH}. "
            "Download it from Kaggle (see README) and place it in data/."
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Copy .env.example → .env and fill it in.")

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Loading samples from {CSV_PATH} …")
    df = load_rare_samples(CSV_PATH, RARE_FINDINGS, max_per_class)
    print(f"Generating reports for {len(df)} samples across {len(RARE_FINDINGS)} classes.\n")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating reports"):
        finding = row["target_finding"]
        image_name = row["Image Index"]

        try:
            report_text = generate_report(client, finding, len(results))
        except anthropic.RateLimitError:
            # Back off and retry once on rate limit
            time.sleep(60)
            report_text = generate_report(client, finding, len(results))

        results.append(
            {
                "image_index": image_name,
                "finding": finding,
                "patient_age": int(row.get("Patient Age", 0)),
                "patient_gender": row.get("Patient Gender", "Unknown"),
                "report": report_text,
            }
        )

        # Brief pause to stay within rate limits
        time.sleep(0.5)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} synthetic reports → {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic radiology reports with Claude.")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=50,
        help="Max samples per rare finding class (default: 50)",
    )
    args = parser.parse_args()
    main(max_per_class=args.max_per_class)
