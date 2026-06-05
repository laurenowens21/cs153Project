
# Synthetic Data Augmentation for Rare Chest X-Ray Finding Classification

## Project Overview

This project investigates the augmentation of synthetic medical imaging data (X-ray images) and 
associated patient reports in a real medical imaging dataset. In this project, I couple the 
synthetic medical images with LLM-generated synthetic radiology reports to introduce symptom, age, gender, 
and associated condition diversity. Focusing on  **Pneumothorax**, and **Emphysema**, we aim to see if the
LLM-generated report and diffusion model-generated chest X-ray images improves binary classification of these conditions 
in the NIH ChestX-ray14 dataset.

---

## Problem and Insight

In public medical imaging datasets, there is severe class imbalance for rare or anatomically 
minor medical findings. The NIH Chest X-ray14 dataset contains over 100,000 images but many 
pathology classes have fewer than 2,000 labeled examples. Training classifiers directly on 
imbalanced data yields poor sensitivity and accuracy on these minority classes, many of which 
are medically significant.

Previous work has investigated the use of synthetic data generation to add more images to these 
datasets or add more demographic diversity to these datasets and have shown that most promising 
results have occurred in models trained on combination of real and synthetic data 
([Stanford Medicine Magazine](https://stanmed.stanford.edu/generative-ai-synthetic-data-promise/), 
[Chen et al](https://pmc.ncbi.nlm.nih.gov/articles/PMC9353344/#R71)). Therefore, I chose to use 
Cloudflare diffusion models to generate medical images and LLMs to generate medical records to 
evaluate if the addition of these synthetic records would result in increased detection of rare 
medical findings.

---

## Technical Work and Experiment

***Image Generation***
For the generation of synthetic chest X-rays, I used Cloudflare Workers AI's Stable Diffusion 
v1.5 to convert each generated report into a corresponding image. A key design decision here 
was to use image-to-image (img2img) generation rather than text-to-image generation. Rather than generating an X-ray from text alone, 
each synthetic image was conditioned on a real reference image randomly sampled from the NIH ChestX-ray14 dataset for the corresponding 
class. Img2img approach preserves the structural properties of real X-rays while still introducing pathology-specific variation through the text prompt.

To construct the image prompt, I extracted the Impression section from each generated report 
and prepended it with a image-specific prefix to anchor the model to the medical imaging 
domain:

"Frontal chest X-ray radiograph, grayscale, DICOM-style medical imaging, high contrast, showing: [Impression text]"

***Record Generation***
A popular topic among our class discussions was the importnance of prompt engineering. I performed two experiments, one with clinician-informed prompts and one with LLM-generated prompts.
Therefore, I experimented with three LLM-prompting strategies using Anthropic Sonnet 4.6 for the generation of textual medical records.: 
        1. Generic Context: Provide the LLM with the condition and tell it to vary a small subset of the features of the records (anatomical size, severity, laterality)
        2. Clinical Context: Further context is added to the Generic Context complete with medical terminology and quantitative axis (anatomical distribution, associated findings) for the introduction of medical context. (Example:  "- Size: small apical (<15% volume loss), moderate (15–60%), or large/tension (>60% with mediastinal shift)"). 
        3. Few-Shot Prompts: The LLM is given an example of a patient record with Pneumothorax or Emphysema and asked to generate a new, different report.


***Classifier Training and Evaluation***
A MobileNetV3-Small network was fine-tuned for each medical condition using a three-class classification setup: Pneumothorax,Emphysema, and No Finding. The real NIH images from data/rare_findings/ were split 80/20 into train and test sets to preserve class proportions across splits. The test set was held fixed across all conditions and only the training set was augmented with synthetic images.
For each prompting strategy, the 50 synthetic images per class were appended to the real training split, and a fresh model was trained  for 10 epochs using Adam (lr=1e-4) with inverse-frequency class weighting to address imbalance. This resulted in four total conditions: one baseline (real images only) and one augmented condition per prompting strategy. Evaluation metrics, per-class and macro-averaged AUC and F1, were computed on the real image test set for all conditions.

---

## Pipeline Description

```
NIH ChestX-ray14 CSV
        │
        ▼
1. Report Generation (src/generate_reports.py)
   └─ Claude (Anthropic API) generates realistic radiology reports
      for each rare finding class → data/synthetic_reports.json

        │
        ▼
2. Image Generation (src/generate_images.py)
   └─ Cloudflare Workers AI (Stable Diffusion) renders chest X-ray
      images conditioned on each synthetic report → data/synthetic_images/

        │
        ▼
3. Classifier Training & Evaluation (src/evaluate.py)
   └─ MobileNetV3 trained in two conditions:
      (A) real images only
      (B) real + synthetic images
      Outputs AUC, F1, confusion matrix → results/

        │
        ▼
4. Image Quality Assessment (src/fid_score.py)
   └─ FID between real and synthetic image sets → results/fid_results.json
```

---

## Setup Instructions

### 1. Clone and enter the repo

```bash
git clone <repo-url>
cd CS153
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your Anthropic API key and you Cloudflare Account ID and API token.
```

### 5. Download NIH ChestX-ray14

Place `Data_Entry_2017.csv` (and optionally the images) in `data/`.
You can download via the [Kaggle API](https://www.kaggle.com/datasets/nih-chest-xrays/data):

```bash
kaggle datasets download -d nih-chest-xrays/data -p data/ --unzip
```

---

## How to Run

Run the steps in order:

```bash
# Step 1 — generate synthetic radiology reports with Claude
python src/generate_reports.py

# Step 2 — generate synthetic X-ray images with Cloudflare Workers AI
python src/generate_images.py

# Step 3 — train classifier and evaluate augmentation benefit
python src/evaluate.py

# Step 4 — compute FID score between real and synthetic images
python src/fid_score.py
```

Results are written to `results/` to be evaluated.

---

## Evaluation Results

Emphysema: Synthetic Image:  <img width="512" height="512" alt="synth_00000013_042" src="https://github.com/user-attachments/assets/61846efe-4c4a-4bb0-aa42-44e84de6f653" />
              Real Image: <img width="1024" height="1024" alt="00000013_042" src="https://github.com/user-attachments/assets/81838a57-9b39-4859-a0b6-435e2aa44d4d" />

Pneumothorax: Real Image: <img width="1024" height="1024" alt="00000013_011" src="https://github.com/user-attachments/assets/cdba89e8-8d77-4b44-a1c1-851db7030ca2" />
              Synthetic Image:   <img width="512" height="512" alt="synth_00000013_011" src="https://github.com/user-attachments/assets/d3c53895-752f-4f84-8348-adbe1c07538d" />


<img width="900" height="750" alt="promptA_confusion_matrix" src="https://github.com/user-attachments/assets/66be3dab-fd58-4417-b7d9-8ed20ec5fa8d" />

V1 refers to the LLM-generated prompts. V2 refers to the LLM-generated prompts with clinician input. 
The above heatmap suggests that the augmented (real + synthetic) data outperforms the baseline (real only) in AUC, suggesting that there is a classification improvement among Pneumothorax, Emphysema, and No Finding. The best record generation prompts are the V1 Few-Shot and the V2 Generic Context. Clinician input was primarily added to the Clinical context and Few-Shot prompts. indicating that if clinician inputs are to be added, they should be appropriately engineered. However, Few-Shot prompting performs consistently well, indicating that the Cloudflare model and LLM learn better with concrete examples rather than general context.

In the image generation, Pneumothorax generates more realistic images than Emphysema. This is likely due to the fact that Pneumothorax has more anatomically rigid boundaries than Emphysema, as collapsed lungs (Pneumothorax) are more noticeable than the small features of Emphysema. This causes some noise in the Emphysema synthetic images. Also, there was about 355 images of Pneumothorax versus 155 for Emphysema.This class imbalance might result in less accurate images for the minority class.

---

## Limitations
 - Generated images lack the ability to diminish quality of the images resulting from technical malfunctions or patient movement during the scan. This is an important facet and ingrained challenge when working with medical data that AI over-optimizes for.
 - Because the NIH Chest X-ray14 dataset is too large to locally host, this project only works with a small subset of this image dataset and uses one single train/test split. Results might not be generalizable without cross-fold validation or a larger training and test set.
 - A GPU was not used for this project. Therefore. compute constraints limit the number of synthetic samples generated per class. 
  - Baseline synthetic report and prompts are generated by a general-purpose text-to-image model not fine-tuned on
  medical data. 
---

## AI Usage Disclosure
I used **Anthropic Claude** (`claude-sonnet-4-6`) and **Anthropic Claude Code Agent** for the code generation for this project, the authorship of the Set Up instructions for the README, generation of the synthetic radiology report text, debugging code

I used **Cloudflare Workers AI**, specifically the Stable Diffusion XL model, to generate synthetic chest X-ray images.

All experimental results,analysis, video script, and authorship of the text portions of the README were performed by myself without the use of AI.

I did not have any collaborators nor did I use any starter code.
