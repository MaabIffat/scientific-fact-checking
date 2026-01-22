# scientific-fact-checking
Code for evaluating and creating SCINLP in scientific fact-checking 


# Scientific Fact-Checking: ACL Claim Extraction Pipeline

This repository provides a reproducible pipeline for automated scientific claim extraction and fact-checking from ACL Anthology papers. The pipeline downloads ACL PDFs and BibTeX metadata, extracts structured research claims using large language models, and produces a final CSV suitable for evaluation (e.g., accuracy, calibration, ECE).

The code is intended for research use in automated fact-checking and scientific NLP.

---

## Repository Structure

scientific-fact-checking/
├── claim_extraction/
│   ├── extract_claims_gpt.py
│   └── prompts.py
├── data_prep/
│   ├── download_acl_pdfs.py
│   ├── download_acl_bibs.py
│   ├── parse_bibs_to_metadata_csv.py
│   └── merge_metadata_and_claims.py
├── evidence_collection/
│   └── collect_evidence_contriever.py
├── evaluation/
│   ├── run_llama_inference.py
│   └── run_hf_inference_unified.py
├── decontextualization_claims/
│   ├── prompts.py
│   └── run_decontextualization.py
├── scripts/
│   └── list_files.py
├── data/                 # local only (ignored)
├── outputs/              # local only (ignored)
├── requirements.txt
├── requirements_evidence.txt
├── README.md
└── LICENSE


---

## Environment Setup

### 1. Create and activate a virtual environment

macOS / Linux:
python3 -m venv venv
source venv/bin/activate

### 2. Install dependencies

pip install -r requirements.txt

### 3. Set your OpenAI API key

Create a `.env` file in the repository root:
touch .env

Add:
OPENAI_API_KEY=YOUR_OPENAI_API_KEY

---

## Data Preparation Pipeline

### Step 1: Download ACL PDFs

Downloads all PDFs linked from an ACL Anthology volume page.

Script:
data_prep/download_acl_pdfs.py

Example:
python data_prep/download_acl_pdfs.py \
  --volume_url https://aclanthology.org/volumes/2024.acl-short/ \
  --output_dir data/raw_pdfs_acl/2024.acl-short \
  --sleep 0.5

Output:
data/raw_pdfs_acl/2024.acl-short/*.pdf

---

### Step 2: Download BibTeX files

Downloads BibTeX metadata for the same ACL volume.

Script:
data_prep/download_acl_bibs.py

Example:
python data_prep/download_acl_bibs.py \
  --volume_url https://aclanthology.org/volumes/2024.acl-short/ \
  --output_dir data/raw_bibs_acl/2024.acl-short \
  --sleep 0.5

Output:
data/raw_bibs_acl/2024.acl-short/*.bib

---

### Step 3: Parse BibTeX files into metadata CSV

Extracts paper titles and author information from BibTeX files.

Script:
data_prep/parse_bibs_to_metadata_csv.py

Example:
python data_prep/parse_bibs_to_metadata_csv.py \
  --bib_root_dir data/raw_bibs_acl \
  --output_csv outputs/papers_info_all_clean.csv

Output CSV columns:
- pdf_file_name
- title
- author

---

## Claim Extraction Using LLMs

This step extracts standalone scientific research claims and research questions, including both Support and Refute labels, along with detailed justifications and keywords.

### Step 4: Extract claims from PDFs

Script:
claim_extraction/extract_claims_gpt.py

Example (test run on a small subset):
python claim_extraction/extract_claims_gpt.py \
  --input_dir data/raw_pdfs_acl/2024.acl-short \
  --output_csv outputs/claims_2024_acl_short.csv \
  --metadata_csv outputs/papers_info_all_clean.csv \
  --model gpt-4o-mini \
  --max_pages 5 \
  --max_words_input 2500 \
  --num_claims 12 \
  --num_reversed 6 \
  --temperature 0.5 \
  --sleep 0.5 \
  --max_pdfs 2

Remove `--max_pdfs` to process the full dataset.

Output CSV columns:
- pdf_file_name
- title
- author
- claim
- label (Support / Refute)
- reason
- keywords

---

## Merge Metadata and Claims (Final Evaluation Input)

Combines metadata and extracted claims into a single CSV used for evaluation.

Script:
data_prep/merge_metadata_and_claims.py

Example:
python data_prep/merge_metadata_and_claims.py \
  --metadata_csv outputs/papers_info_all_clean.csv \
  --claims_csv outputs/claims_2024_acl_short.csv \
  --output_csv outputs/final_acl_2024_short_claims.csv

Final output:
outputs/final_acl_2024_short_claims.csv

This file is used as input for downstream evaluation and analysis.

---

## Recommended .gitignore

venv/
.env

data/raw_pdfs_acl/
data/raw_bibs_acl/
*.pdf
*.bib

outputs/
!outputs/final_acl_2024_short_claims.csv

------------------------------------------------------

## Notes

- Files such as *.0.pdf correspond to proceedings/front-matter and may not contain authors.
- If claim parsing fails, reduce input size or inspect raw model outputs.

------------------------------------------------------

Evidence Collection (Contriever-based)

This step enriches extracted claims with evidence sentences from the original PDFs using the Contriever dense retriever.

For each claim, the script:
	•	locates the corresponding PDF
	•	extracts text from the first pages
	•	retrieves top-3 most relevant sentences
	•	optionally fills missing title, abstract, and keywords

Setup (Evidence Only)
python -m venv venv_evidence
source venv_evidence/bin/activate
pip install -r requirements_evidence.txt

Run Evidence Collection
python evidence_collection/collect_evidence_contriever.py \
  --input_csv outputs/final_acl_2024_short_claims.csv \
  --output_csv outputs/final_acl_2024_short_claims_with_evidence.csv \
  --pdf_root_dir data/raw_pdfs_acl \
  --max_rows 5

---------------------------------------------------------

Evaluation (Veracity & Time-Shift Analysis)

This repository evaluates scientific NLP claims using large language models (LLaMA and Qwen).
Each claim is assessed for:
	•	Veracity (True / False)
	•	Temporal status (Time-Shift):
	•	Advanced
	•	Outdated
	•	Still holds true (even if not current)

The evaluation follows the exact structured JSON prompt described in the paper, including detailed justifications for both veracity and temporal labels.

Example to run evaluation script
python evaluation/run_hf_inference_unified.py \
  --family qwen \
  --model 7b \
  --setting claim_title_evidence_abstract_keywords \
  --input_csv outputs/final_acl_2024_short_claims_with_evidence.csv \
  --output_json outputs/model_runs/qwen7_time_shift.json

Arguments
	•	--family : llama or qwen
	•	--model  :
	•	LLaMA → 8b, 70b
	•	Qwen  → 7b, 72b
	•	--setting : One of the 6 evaluation settings:
	•	claim
	•	claim_title
	•	claim_evidence
	•	claim_abstract
	•	claim_title_abstract_keywords
	•	claim_title_evidence_abstract_keywords

Outputs

Each output entry includes:
	•	Veracity
	•	Justification
	•	Time_Label
	•	Justification_Time_Label
	•	Original claim metadata (year, title, evidence, etc.)

Decontextualization and Ambiguity Annotation

decontextualization_claims/ folder contains scripts to (1) generate decontextualized (standalone) claims and (2) extract ambiguity-related subject and disambiguation criteria using GPT with k-shot prompting.

Example to run
python decontextualization_claims/run_decontextualization.py \
  --input_csv /path/to/input.csv \
  --n 50 \
  --model gpt-4

OUTPUTS

	•	DECONTEXTUALIZED_CLAIM
	•	SUBJECT
	•	DISAMBIGUATION_CRITERIA
