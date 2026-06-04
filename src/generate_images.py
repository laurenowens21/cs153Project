"""
generate_images.py

Reads data/synthetic_reports.json and calls Cloudflare Workers AI's
Stable Diffusion endpoint to generate a synthetic chest X-ray image for each
report. Images are saved as grayscale PNGs under data/synthetic_images/<finding>/.

Usage:
    python src/generate_images.py [--input data/synthetic_reports.json]
                                  [--output-dir data/synthetic_images]
                                  [--size 512]
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_INPUT = DATA_DIR / "synthetic_reports.json"
DEFAULT_OUTPUT_DIR = DATA_DIR / "synthetic_images"

# Cloudflare Workers AI model for image generation
CF_MODEL = "@cf/stabilityai/stable-diffusion-xl-base-1.0"

# Prefix added to every prompt to steer the model toward X-ray style output
IMAGE_PROMPT_PREFIX = (
    "Frontal chest X-ray radiograph, grayscale, medical imaging, high resolution, "
    "DICOM-style, showing: "
)

# Negative prompt to reduce photographic/cartoon artifacts
NEGATIVE_PROMPT = (
    "color photograph, illustration, cartoon, painting, blurry, text, watermark, "
    "person, face, body, skin"
)


def build_image_prompt(report: str, finding: str) -> str:
    """
    Convert a radiology report text into a concise image generation prompt.
    Uses only the IMPRESSION section if present, otherwise first 200 chars.
    """
    impression = report
    if "IMPRESSION:" in report.upper():
        impression = report.upper().split("IMPRESSION:")[-1].strip()
        # Restore original case from the original report
        idx = report.upper().find("IMPRESSION:")
        impression = report[idx + len("IMPRESSION:"):].strip()

    # Truncate to keep prompts concise
    short = impression[:300].rstrip()
    return f"{IMAGE_PROMPT_PREFIX}{finding}, {short}"


def call_cloudflare_sd(
    prompt: str,
    account_id: str,
    api_token: str,
    image_size: int = 512,
) -> bytes:
    """
    Call the Cloudflare Workers AI text-to-image endpoint.

    Returns raw PNG bytes on success.
    Raises requests.HTTPError on non-2xx response.
    """
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{CF_MODEL}"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "num_steps": 20,
        "width": image_size,
        "height": image_size,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.content


def save_image(image_bytes: bytes, output_path: Path) -> None:
    """Save raw bytes as a grayscale PNG, converting from RGB if needed."""
    from io import BytesIO

    img = Image.open(BytesIO(image_bytes)).convert("L")  # L = grayscale
    img.save(output_path, format="PNG")


def main(input_path: Path, output_dir: Path, image_size: int = 512) -> None:
    """
    Main entry point. Iterates over synthetic_reports.json and generates
    one image per report entry.

    Args:
        input_path:  Path to synthetic_reports.json.
        output_dir:  Root directory for output images (sub-dirs per finding).
        image_size:  Width and height in pixels for generated images.
    """
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")

    if not account_id or not api_token:
        raise EnvironmentError(
            "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must be set in .env"
        )

    if not input_path.exists():
        raise FileNotFoundError(
            f"synthetic_reports.json not found at {input_path}. "
            "Run src/generate_reports.py first."
        )

    with open(input_path) as f:
        reports = json.load(f)

    print(f"Loaded {len(reports)} reports. Generating images …\n")

    failed = []
    for entry in tqdm(reports, desc="Generating images"):
        finding = entry["finding"]
        image_name = Path(entry["image_index"]).stem  # strip .png if present

        finding_dir = output_dir / finding
        finding_dir.mkdir(parents=True, exist_ok=True)

        out_path = finding_dir / f"synth_{image_name}.png"
        if out_path.exists():
            continue  # skip already-generated images to allow resumption

        prompt = build_image_prompt(entry["report"], finding)

        try:
            image_bytes = call_cloudflare_sd(prompt, account_id, api_token, image_size)
            save_image(image_bytes, out_path)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            tqdm.write(f"  HTTP {status} for {image_name} — skipping")
            failed.append({"image_index": entry["image_index"], "error": str(e)})
            if status == 429:
                # Rate limited: back off before continuing
                time.sleep(10)
            continue

        time.sleep(0.2)  # conservative pacing

    total = len(reports)
    succeeded = total - len(failed)
    print(f"\nDone: {succeeded}/{total} images saved to {output_dir}")

    if failed:
        failure_log = output_dir / "failed.json"
        with open(failure_log, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"Failures logged to {failure_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic X-ray images via Cloudflare AI.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to synthetic_reports.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory for generated images",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Image width/height in pixels (default: 512)",
    )
    args = parser.parse_args()
    main(input_path=args.input, output_dir=args.output_dir, image_size=args.size)
