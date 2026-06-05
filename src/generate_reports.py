"""
generate_reports.py

Generates synthetic chest X-ray images for Pneumothorax and Emphysema using
three distinct prompt strategies. For each strategy the script:

  1. Calls Claude to produce 50 synthetic radiology reports per class.
  2. Calls Cloudflare Workers AI (img2img) to render one image per report,
     conditioned on a randomly selected real reference X-ray.

Outputs
-------
  data/synthetic_reports_promptA.json   — reports, generic strategy
  data/synthetic_reports_promptB.json   — reports, clinically detailed strategy
  data/synthetic_reports_promptC.json   — reports, few-shot strategy
  data/synthetic_images_promptA/<finding>/*.png
  data/synthetic_images_promptB/<finding>/*.png
  data/synthetic_images_promptC/<finding>/*.png

Strategies
----------
  A — Generic:             Short, one-sentence description of the finding.
  B — Clinically Detailed: Rich anatomical context, severity grades, demographics.
  C — Few-shot:            Two labelled example reports included in the prompt.

Usage:
    python src/generate_reports.py [--n-per-class 50] [--strategy A] [--strategy B]
"""

import argparse
import io
import json
import os
import random
import time
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
REAL_IMAGE_DIR = DATA_DIR / "rare_findings"

FINDINGS = ["Pneumothorax", "Emphysema"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
CF_MODEL = "@cf/runwayml/stable-diffusion-v1-5-img2img"
IMAGE_SIZE = 512

# ── System prompt (shared by all strategies) ──────────────────────────────────

SYSTEM_PROMPT = (
    "You are a board-certified radiologist writing official chest X-ray radiology reports "
    "for an academic medical center. Your reports are factual, use standard radiological "
    "terminology, and follow this exact structure:\n"
    "CLINICAL INDICATION:\n"
    "TECHNIQUE:\n"
    "FINDINGS:\n"
    "IMPRESSION:\n"
    "Do not include patient identifiers. Return only the report text."
)

# ── Per-finding prompt context used by strategies A and B ─────────────────────

GENERIC_CONTEXT = {
    "Pneumothorax": (
        "a pneumothorax visible on chest X-ray. "
        "Vary the size (small, moderate, large) and laterality across reports."
    ),
    "Emphysema": (
        "pulmonary emphysema visible on chest X-ray. "
        "Vary the severity (mild, moderate, severe) across reports."
    ),
}

CLINICAL_CONTEXT = {
    "Pneumothorax": (
        "a pneumothorax. Vary the following realistically across reports:\n"
        "- Size: small apical (<15% volume loss), moderate (15–60%), or large/tension "
        "(>60% with mediastinal shift)\n"
        "- Laterality: left or right\n"
        "- Associated findings: subcutaneous emphysema, rib fractures, pleural thickening, "
        "prior chest tube tracts, or contralateral lung findings\n"
        "- Patient demographics: age 18–75, either sex, varied clinical indication "
        "(spontaneous, traumatic, iatrogenic, or underlying lung disease)\n"
        "- Technique: portable AP, upright PA, or lateral decubitus as appropriate"
    ),
    "Emphysema": (
        "pulmonary emphysema (COPD). Vary the following realistically across reports:\n"
        "- Severity: mild (subtle hyperinflation), moderate (flattened diaphragms, "
        "increased AP diameter), or severe (bullous disease, attenuated vascularity, "
        "cor pulmonale)\n"
        "- Distribution: panlobular vs centrilobular, upper vs lower lobe predominance\n"
        "- Associated findings: pulmonary hypertension signs, flattened cardiac silhouette, "
        "spontaneous pneumothorax, or superimposed infection\n"
        "- Patient demographics: age 50–80, heavy smoker history, varied clinical indication\n"
        "- Technique: PA and lateral, or portable AP"
    ),
}

# ── Few-shot examples (Strategy C) ────────────────────────────────────────────
# Two full example reports are included in the prompt to demonstrate format and
# clinical depth. One example per finding class is shown regardless of the
# current target finding to maximise diversity.

FEW_SHOT_EXAMPLES = {
    "Pneumothorax": """\
Here are two example reports to guide your format and clinical depth.

--- EXAMPLE 1 (Pneumothorax) ---
CLINICAL INDICATION: 28-year-old male with sudden-onset left pleuritic chest pain and dyspnea. Rule out pneumothorax.

TECHNIQUE: Single anteroposterior portable chest radiograph obtained in the upright position.

FINDINGS:
There is a moderate left-sided pneumothorax with a visible visceral pleural line approximately 3.5 cm from the lateral chest wall at the mid-lung level. Partial collapse of the left lower lobe is present. The trachea and mediastinum remain midline without evidence of tension physiology. The right lung is clear without focal consolidation, atelectasis, or pleural effusion. The cardiac silhouette is normal in size and contour. No rib fractures identified. Bony thorax otherwise intact.

IMPRESSION:
1. Moderate left-sided pneumothorax with partial left lower lobe atelectasis.
2. No mediastinal shift. No tension pneumothorax.
Recommend thoracic surgery consultation.

--- EXAMPLE 2 (Pneumothorax) ---
CLINICAL INDICATION: 55-year-old female with COPD, presenting with acute worsening dyspnea. Prior left-sided pleurodesis.

TECHNIQUE: PA and lateral chest radiographs.

FINDINGS:
Small right apical pneumothorax measuring less than 1 cm at the apex with a thin pleural line. No significant volume loss. The left hemithorax shows diffuse pleural thickening consistent with prior pleurodesis and is unchanged from prior exam. Bilateral hyperinflation consistent with underlying emphysema. Mild flattening of the hemidiaphragms bilaterally. No new focal consolidation. Cardiac silhouette is within normal limits.

IMPRESSION:
1. Small right apical pneumothorax, likely spontaneous in the setting of underlying emphysema.
2. Stable left pleural thickening post-pleurodesis.
3. Bilateral emphysema, unchanged.
---

Now generate a new, DIFFERENT report for a patient with a pneumothorax. Vary the size, laterality, and clinical context from the examples above.""",

    "Emphysema": """\
Here are two example reports to guide your format and clinical depth.

--- EXAMPLE 1 (Emphysema) ---
CLINICAL INDICATION: 67-year-old male with 45 pack-year smoking history presenting with progressive exertional dyspnea and chronic productive cough.

TECHNIQUE: PA and lateral chest radiographs obtained in the upright position.

FINDINGS:
The lungs are markedly hyperinflated bilaterally with flattening of both hemidiaphragms, best appreciated on the lateral projection. There is an increased AP diameter consistent with air trapping. The pulmonary vascularity is attenuated peripherally. Several bilateral upper lobe bullae are present, the largest measuring approximately 3 cm on the right. The cardiac silhouette appears elongated and narrow in its vertical dimension. No focal consolidation or pleural effusion is identified. The mediastinum is not widened. Bony thorax shows no acute fracture.

IMPRESSION:
1. Severe pulmonary emphysema with bilateral hyperinflation, bilateral diaphragmatic flattening, and upper lobe bullous disease, right greater than left.
2. Findings consistent with advanced COPD.

--- EXAMPLE 2 (Emphysema) ---
CLINICAL INDICATION: 72-year-old female with known COPD, presenting for annual follow-up. Chronic dyspnea on exertion.

TECHNIQUE: PA chest radiograph.

FINDINGS:
Mild bilateral pulmonary hyperinflation with mild flattening of the hemidiaphragms. The AP diameter is mildly increased on clinical assessment. Pulmonary vascularity is mildly decreased peripherally. No bullae identified. No focal consolidation, pneumothorax, or pleural effusion. The cardiac silhouette is at the upper limits of normal. Aortic knuckle is prominent, possibly related to mild aortic unfolding. No acute bony abnormality.

IMPRESSION:
1. Mild-to-moderate pulmonary emphysema.
2. No acute cardiopulmonary process.
---

Now generate a new, DIFFERENT report for a patient with pulmonary emphysema. Vary the severity, associated findings, and clinical context from the examples above.""",
}

# ── Prompt builders ───────────────────────────────────────────────────────────


def build_prompt_A(finding: str, index: int) -> str:
    """Strategy A — Generic: minimal one-sentence description."""
    context = GENERIC_CONTEXT[finding]
    return (
        f"Generate a realistic chest X-ray radiology report for a patient with {context} "
        f"Sample index: {index}. Return only the report text."
    )


def build_prompt_B(finding: str, index: int) -> str:
    """Strategy B — Clinically Detailed: rich anatomical and demographic context."""
    context = CLINICAL_CONTEXT[finding]
    return (
        f"Generate a realistic chest X-ray radiology report for a patient with {context}\n\n"
        f"Sample index: {index}. Return only the report text."
    )


def build_prompt_C(finding: str, index: int) -> str:
    """Strategy C — Few-shot: two in-context example reports precede the request."""
    return FEW_SHOT_EXAMPLES[finding] + f"\n\nSample index: {index}."


STRATEGIES: dict[str, dict] = {
    "A": {"label": "Generic",              "builder": build_prompt_A},
    "B": {"label": "Clinically Detailed",  "builder": build_prompt_B},
    "C": {"label": "Few-shot",             "builder": build_prompt_C},
}

# ── Report generation ─────────────────────────────────────────────────────────


def generate_report(
    client: anthropic.Anthropic,
    finding: str,
    strategy: str,
    index: int,
) -> str:
    """Call Claude with the chosen strategy prompt and return the report text."""
    builder = STRATEGIES[strategy]["builder"]
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=700,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": builder(finding, index)}],
    )
    return response.content[0].text.strip()


def generate_reports_for_strategy(
    client: anthropic.Anthropic,
    strategy: str,
    n_per_class: int,
) -> list[dict]:
    """
    Generate n_per_class reports for each finding using the given strategy.
    Returns list of dicts with keys: image_index, finding, strategy, report.
    Loads any prior progress from the output JSON to allow resumption.
    """
    out_path = DATA_DIR / f"synthetic_reports_prompt{strategy}.json"

    existing: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path) as f:
            for entry in json.load(f):
                existing[entry["image_index"]] = entry

    results = list(existing.values())
    label = STRATEGIES[strategy]["label"]

    for finding in FINDINGS:
        needed = n_per_class - sum(1 for r in results if r["finding"] == finding)
        if needed <= 0:
            print(f"  [{strategy}] {finding}: already complete ({n_per_class} reports)")
            continue

        print(f"  [{strategy}] {finding}: generating {needed} reports …")
        start_idx = sum(1 for r in results if r["finding"] == finding)

        for i in tqdm(range(needed), desc=f"    {finding}", leave=False):
            idx = start_idx + i
            img_key = f"synth_{strategy}_{finding}_{idx:04d}.png"

            try:
                report_text = generate_report(client, finding, strategy, idx)
            except anthropic.RateLimitError:
                tqdm.write("  Rate limited — waiting 60 s …")
                time.sleep(60)
                report_text = generate_report(client, finding, strategy, idx)

            results.append({
                "image_index": img_key,
                "finding": finding,
                "strategy": strategy,
                "report": report_text,
            })

            # Save after every entry for crash resilience
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

            time.sleep(0.4)

    print(f"  [{strategy}] Reports saved → {out_path.name}")
    return results


# ── Image generation ──────────────────────────────────────────────────────────


def build_ref_index(real_dir: Path) -> dict[str, list[Path]]:
    """Map each finding to its list of real reference image paths."""
    return {
        finding: sorted(
            p for p in (real_dir / finding).iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for finding in FINDINGS
        if (real_dir / finding).exists()
    }


def extract_impression(report: str, finding: str) -> str:
    """Pull IMPRESSION section for use as the image prompt; fall back to full report."""
    upper = report.upper()
    if "IMPRESSION:" in upper:
        idx = upper.find("IMPRESSION:")
        impression = report[idx + len("IMPRESSION:"):].strip()
    else:
        impression = report.strip()

    prefix = (
        "Frontal chest X-ray radiograph, grayscale, DICOM-style medical imaging, "
        "high contrast, showing: "
    )
    return f"{prefix}{finding}, {impression[:300].rstrip()}"


def encode_image(path: Path, size: int) -> list[int]:
    """Resize image to size×size and return raw PNG bytes as an integer list."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return list(buf.getvalue())


def call_cloudflare(
    prompt: str,
    image_ints: list[int],
    account_id: str,
    api_token: str,
) -> bytes:
    """Call the Cloudflare img2img endpoint and return raw PNG bytes."""
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/ai/run/{CF_MODEL}"
    )
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"},
        json={
            "prompt": prompt,
            "negative_prompt": (
                "color photo, illustration, cartoon, text, watermark, face, person, "
                "blurry, low quality, artifacts"
            ),
            "image": image_ints,
            "strength": 0.65,
            "num_steps": 20,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.content


def generate_images_for_strategy(
    reports: list[dict],
    strategy: str,
    ref_index: dict[str, list[Path]],
    account_id: str,
    api_token: str,
) -> None:
    """
    For each report entry generate one synthetic image and save it under
    data/synthetic_images_prompt{strategy}/<finding>/.
    Already-existing files are skipped so the run is resumable.
    """
    out_root = DATA_DIR / f"synthetic_images_prompt{strategy}"
    failed = []

    for entry in tqdm(reports, desc=f"  [{strategy}] Images"):
        finding = entry["finding"]
        img_name = entry["image_index"]

        out_dir = out_root / finding
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / img_name

        if out_path.exists():
            continue

        refs = ref_index.get(finding, [])
        if not refs:
            tqdm.write(f"  [SKIP] No reference images for {finding}")
            failed.append(img_name)
            continue

        ref_path = random.choice(refs)
        prompt = extract_impression(entry["report"], finding)

        try:
            image_bytes = call_cloudflare(
                prompt, encode_image(ref_path, IMAGE_SIZE), account_id, api_token
            )
            img = Image.open(io.BytesIO(image_bytes)).convert("L")
            img.save(out_path, format="PNG")
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            tqdm.write(f"  HTTP {status} — skipping {img_name}")
            failed.append(img_name)
            if status == 429:
                time.sleep(15)
            continue

        time.sleep(0.25)

    if failed:
        fail_path = out_root / "failed.json"
        with open(fail_path, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"  [{strategy}] {len(failed)} failures logged → {fail_path.name}")

    total = len(reports)
    print(f"  [{strategy}] Images saved → {out_root.name}/")


# ── Main ──────────────────────────────────────────────────────────────────────


def main(strategies: list[str], n_per_class: int) -> None:
    """
    Run report generation then image generation for each requested strategy.

    Args:
        strategies:  List of strategy keys to run, e.g. ["A", "B", "C"].
        n_per_class: Reports (and images) to generate per finding class.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    cf_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    cf_account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")

    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in .env")
    if not cf_token or not cf_account:
        raise EnvironmentError("CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID must be set in .env")

    if not REAL_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Real image directory not found: {REAL_IMAGE_DIR}. "
            "Run scripts/filter_images.py first."
        )

    claude = anthropic.Anthropic(api_key=api_key)
    ref_index = build_ref_index(REAL_IMAGE_DIR)

    for finding, paths in ref_index.items():
        print(f"  Reference images — {finding}: {len(paths)}")
    print()

    for strategy in strategies:
        if strategy not in STRATEGIES:
            print(f"[WARN] Unknown strategy '{strategy}', skipping.")
            continue

        label = STRATEGIES[strategy]["label"]
        print(f"{'=' * 60}")
        print(f"  Strategy {strategy}: {label}")
        print(f"{'=' * 60}")

        # Step 1: generate reports
        reports = generate_reports_for_strategy(claude, strategy, n_per_class)

        # Step 2: generate images from those reports
        generate_images_for_strategy(reports, strategy, ref_index, cf_account, cf_token)

        print()

    print("Done. Summary of outputs:")
    for strategy in strategies:
        for finding in FINDINGS:
            img_dir = DATA_DIR / f"synthetic_images_prompt{strategy}" / finding
            n = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
            print(f"  prompt{strategy}/{finding}: {n} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic X-ray reports and images for all prompt strategies."
    )
    parser.add_argument(
        "--strategy",
        action="append",
        dest="strategies",
        choices=list(STRATEGIES.keys()),
        help="Strategy to run (repeat for multiple, default: all three)",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=50,
        help="Reports/images to generate per finding class per strategy (default: 50)",
    )
    args = parser.parse_args()
    chosen = args.strategies if args.strategies else list(STRATEGIES.keys())
    main(strategies=chosen, n_per_class=args.n_per_class)
