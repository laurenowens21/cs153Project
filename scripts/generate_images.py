"""
generate_images.py

Generates synthetic chest X-ray images for each entry in
data/synthetic_reports.json using Cloudflare Workers AI's img2img endpoint.

Instead of pure text-to-image generation, this script conditions each output on
a real reference image from data/rare_findings/<finding>/ at a moderate strength.
This preserves the structural characteristics of real X-rays (aspect ratio,
grayscale distribution, anatomical layout) while producing novel, varied images.

Outputs are saved as grayscale PNGs to data/synthetic_images/<finding>/.
The script is resumable: already-generated images are skipped on re-run.

Usage:
    python scripts/generate_images.py [--strength 0.65] [--size 512]
"""

import argparse
import io
import os
import random
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
REPORTS_PATH = ROOT / "data" / "synthetic_reports.json"
REAL_IMAGE_DIR = ROOT / "data" / "rare_findings"
OUTPUT_DIR = ROOT / "data" / "synthetic_images"

# img2img model — conditions generation on an input image + prompt.
# Produces outputs that preserve structural X-ray characteristics from the
# reference while varying pathology appearance per the report-derived prompt.
CF_MODEL = "@cf/runwayml/stable-diffusion-v1-5-img2img"

IMAGE_SIZE = 512

# Prompt prefix that steers output toward radiograph style
PROMPT_PREFIX = (
    "Frontal chest X-ray radiograph, grayscale, DICOM-style medical imaging, "
    "high contrast, showing: "
)

NEGATIVE_PROMPT = (
    "color photo, illustration, cartoon, text, watermark, logo, face, person, "
    "blurry, low quality, artifacts, border"
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def extract_prompt_from_report(report: str, finding: str) -> str:
    """
    Pull the IMPRESSION section from a radiology report to use as the
    image prompt. Falls back to the full report if IMPRESSION is absent.
    Truncated to 350 chars to stay within model limits.
    """
    upper = report.upper()
    if "IMPRESSION:" in upper:
        idx = upper.find("IMPRESSION:")
        impression = report[idx + len("IMPRESSION:"):].strip()
    else:
        impression = report.strip()

    short = impression[:350].rstrip()
    return f"{PROMPT_PREFIX}{finding}, {short}"


def encode_image_bytes(image_path: Path, size: int) -> list[int]:
    """
    Open an image, resize to size×size, save as PNG, and return the raw file
    bytes as a flat list of integers.

    Cloudflare Workers AI img2img expects the image as an integer array of
    PNG file bytes — not base64, not raw pixel values.
    """
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return list(buf.getvalue())


def call_img2img(
    prompt: str,
    reference_bytes: list[int],
    account_id: str,
    api_token: str,
    strength: float,
    size: int,
) -> bytes:
    """
    Call the Cloudflare Workers AI img2img endpoint.

    Args:
        prompt:           Text prompt derived from the synthetic report.
        reference_bytes:  PNG file bytes as a flat list of integers.
                          Cloudflare requires this format — not base64.
        account_id:       Cloudflare account ID.
        api_token:        Cloudflare API token.
        strength:         How much to deviate from the reference (0 = copy,
                          1 = ignore reference). 0.6–0.7 keeps X-ray structure
                          while allowing pathology variation.
        size:             Output width and height in pixels.

    Returns raw PNG bytes on success. Raises requests.HTTPError otherwise.
    """
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/ai/run/{CF_MODEL}"
    )
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "image": reference_bytes,
        "strength": strength,
        "num_steps": 20,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.content


def save_grayscale_png(image_bytes: bytes, output_path: Path) -> None:
    """Decode raw image bytes and save as a grayscale PNG."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img.save(output_path, format="PNG")


def build_reference_index(real_image_dir: Path, findings: list[str]) -> dict[str, list[Path]]:
    """
    Build a dict mapping each finding to the list of real reference image paths.
    These are sampled randomly during generation so each synthetic image gets
    a different structural reference.
    """
    index = {}
    for finding in findings:
        class_dir = real_image_dir / finding
        if not class_dir.exists():
            index[finding] = []
            continue
        index[finding] = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return index


def main(strength: float = 0.65, size: int = IMAGE_SIZE) -> None:
    """
    Main entry point. For each entry in synthetic_reports.json, selects a
    random real reference image from the same finding class, conditions the
    img2img model on it, and saves the output to data/synthetic_images/.

    Args:
        strength: img2img denoising strength (0–1). Higher = more deviation
                  from the reference image. 0.65 preserves X-ray anatomy
                  while allowing visible pathology variation.
        size:     Output image width and height in pixels.
    """
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account_id or not api_token:
        raise EnvironmentError(
            "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must be set in .env"
        )

    if not REPORTS_PATH.exists():
        raise FileNotFoundError(
            f"synthetic_reports.json not found at {REPORTS_PATH}. "
            "Run scripts/generate_reports.py first."
        )

    import json
    with open(REPORTS_PATH) as f:
        reports = json.load(f)

    findings = list({r["finding"] for r in reports})
    ref_index = build_reference_index(REAL_IMAGE_DIR, findings)

    for finding in findings:
        n = len(ref_index.get(finding, []))
        print(f"  {finding}: {n} reference images available")

    print(f"\n{len(reports)} reports loaded. Generating images (strength={strength}) …\n")

    failed = []
    skipped = 0

    for entry in tqdm(reports, desc="Generating images"):
        finding = entry["finding"]
        stem = Path(entry["image_index"]).stem

        out_dir = OUTPUT_DIR / finding
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"synth_{stem}.png"

        if out_path.exists():
            skipped += 1
            continue

        references = ref_index.get(finding, [])
        if not references:
            tqdm.write(f"  [SKIP] No reference images for {finding}")
            failed.append({"image_index": entry["image_index"], "error": "no reference images"})
            continue

        # Pick a random reference so each synthetic image has a distinct basis
        ref_path = random.choice(references)

        try:
            ref_bytes = encode_image_bytes(ref_path, size)
            prompt = extract_prompt_from_report(entry["report"], finding)
            image_bytes = call_img2img(prompt, ref_bytes, account_id, api_token, strength, size)
            save_grayscale_png(image_bytes, out_path)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            tqdm.write(f"  HTTP {status} for {entry['image_index']} — skipping")
            failed.append({"image_index": entry["image_index"], "error": str(e)})
            if status == 429:
                time.sleep(15)
            continue

        time.sleep(0.3)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(reports)
    succeeded = total - skipped - len(failed)
    print(f"\nResults:")
    print(f"  Generated:  {succeeded}")
    print(f"  Skipped (already existed): {skipped}")
    print(f"  Failed:     {len(failed)}")
    print(f"  Output dir: {OUTPUT_DIR.resolve()}")

    if failed:
        failure_log = OUTPUT_DIR / "failed.json"
        import json
        with open(failure_log, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"  Failures logged → {failure_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic chest X-ray images conditioned on real references."
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.65,
        help=(
            "img2img denoising strength 0–1 (default: 0.65). "
            "Lower = closer to reference structure; higher = more novel."
        ),
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Output image width and height in pixels (default: 512)",
    )
    args = parser.parse_args()
    main(strength=args.strength, size=args.size)
