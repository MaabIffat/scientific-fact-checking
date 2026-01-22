import os
import re
import argparse
from collections import Counter
from typing import List, Dict

import pandas as pd
import fitz  # PyMuPDF
import nltk
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import stopwords
from nltk.util import ngrams


# ----------------------------
# Utilities
# ----------------------------
def get_field(row: pd.Series, *candidates: str) -> str:
    for c in candidates:
        if c in row and pd.notna(row[c]):
            return str(row[c])
    return ""


def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


def build_pdf_index(pdf_root_dir: str) -> Dict[str, str]:
    """
    Build mapping: lowercased filename -> full path
    Searches recursively under pdf_root_dir.
    """
    idx = {}
    for dirpath, _, filenames in os.walk(pdf_root_dir):
        for fn in filenames:
            if fn.lower().endswith(".pdf"):
                idx[fn.strip().lower()] = os.path.join(dirpath, fn)
    return idx


# ----------------------------
# PDF extraction
# ----------------------------
def extract_text_from_pdf(pdf_path: str, max_pages: int = 5) -> List[str]:
    try:
        doc = fitz.open(pdf_path)
        pages = [page.get_text() for page in doc[:max_pages]]
        doc.close()
        return pages
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
        return []


def extract_title_from_first_page(text_pages: List[str], pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        first_page = doc[0]
        blocks = first_page.get_text("dict")["blocks"]
        doc.close()

        candidates = []
        for block in blocks:
            if "lines" not in block:
                continue

            block_text = ""
            max_font = 0
            bold = False

            for line in block["lines"]:
                for span in line["spans"]:
                    block_text += span["text"] + " "
                    max_font = max(max_font, span.get("size", 0))
                    if span.get("flags", 0) & 2:
                        bold = True

            if re.search(r"\babstract\b", block_text, re.IGNORECASE):
                break

            if len(block_text.strip()) > 6 and max_font > 12:
                candidates.append((block_text.strip(), max_font, bold))

        if candidates:
            candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
            title = candidates[0][0]
            if len(candidates) > 1 and abs(candidates[0][1] - candidates[1][1]) < 1.5:
                title += " " + candidates[1][0]

            title = re.sub(r"\n+", " ", title)
            title = re.sub(
                r"proceedings of the .*?association for computational linguistics.*",
                "",
                title,
                flags=re.I,
            )
            return title.strip()

        # fallback
        lines = [l.strip() for l in (text_pages[0] if text_pages else "").split("\n") if l.strip()]
        return lines[0] if lines else ""

    except Exception as e:
        print(f"❌ Title extraction error: {e}")
        return ""


def extract_abstract_from_text(text_pages: List[str]) -> str:
    joined = "\n".join(text_pages[:3])
    pat = re.compile(
        r"(?is)abstract[\s:\-\.]*([\s\S]*?)(?:\n{1,2}(?:\d{0,1}\s*introduction|introduction)\b|\n{2,})"
    )
    m = pat.search(joined)
    return m.group(1).strip() if m else ""


def extract_top10_keywords(text: str, stop_words: set) -> str:
    sents = nltk.sent_tokenize(text)
    words = [w.lower() for s in sents for w in nltk.word_tokenize(s)]
    filtered = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 1]

    phrases = []
    for n in [2, 3]:
        phrases += [" ".join(g) for g in ngrams(filtered, n)]

    counts = Counter(phrases + filtered)
    top = counts.most_common(10)
    return ", ".join([w for w, _ in top])


# ----------------------------
# Contriever similarity
# ----------------------------
@torch.no_grad()
def find_top_3_sentences(
    claim: str,
    sentences: List[str],
    tokenizer,
    model,
    device,
    batch_size: int = 100,
) -> List[str]:
    if not sentences:
        return ["", "", ""]

    claim_tok = tokenizer(claim, return_tensors="pt", truncation=True, max_length=512).to(device)
    claim_emb = model(**claim_tok).pooler_output

    sims = []
    for i in range(0, len(sentences), batch_size):
        batch = tokenizer(
            sentences[i:i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        emb = model(**batch).pooler_output
        sims.extend(torch.cosine_similarity(claim_emb, emb).cpu().tolist())

    ranked = sorted(zip(sentences, sims), key=lambda x: x[1], reverse=True)
    top = [s for s, _ in ranked[:3]]
    return top + [""] * (3 - len(top))


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Collect evidence using Contriever (top-3 relevant sentences) from PDFs.")
    parser.add_argument("--input_csv", required=True, help="Your final merged CSV (has File Name, Year, Claim, etc.)")
    parser.add_argument("--output_csv", required=True, help="Output CSV with Evidence column added")
    parser.add_argument("--pdf_root_dir", required=True, help="Root directory containing PDFs (script will search recursively)")
    parser.add_argument("--max_pages", type=int, default=5)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    ensure_nltk()
    stop_words = set(stopwords.words("english"))

    if not os.path.exists(args.input_csv):
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    if args.max_rows > 0:
        df = df.head(args.max_rows).reset_index(drop=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    print(f"🔎 Indexing PDFs under: {args.pdf_root_dir}")
    pdf_index = build_pdf_index(args.pdf_root_dir)
    print(f"✅ Found {len(pdf_index)} PDFs.")

    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever").to(device)
    model.eval()
    print("✅ Contriever loaded.")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    # Output columns (keep your schema + add Evidence)
    out_cols = [
        "Index",
        "Year",
        "author",
        "Reason",
        "File Name",
        "Claim",
        "Support or Refute Label",
        "title",
        "Abstract",
        "Keywords Extracted",
        "Evidence",
    ]
    pd.DataFrame(columns=out_cols).to_csv(args.output_csv, index=False)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        pdf_file = get_field(row, "File Name", "pdf_file_name", "File_Name").strip()
        claim = get_field(row, "Claim", "claim").strip()
        year = get_field(row, "Year", "year").strip()

        author = get_field(row, "author", "Author")
        reason = get_field(row, "Reason", "reason")
        label = get_field(row, "Support or Refute Label", "Ground_Label", "label")
        index_ = get_field(row, "Index", "index")

        title = get_field(row, "title", "Title")
        abstract = get_field(row, "Abstract")
        keywords = get_field(row, "Keywords Extracted", "Keywords_Extracted", "keywords")
        evidence = ""

        pdf_path = pdf_index.get(pdf_file.lower(), "")

        if pdf_path and claim:
            pages = extract_text_from_pdf(pdf_path, args.max_pages)
            if pages:
                if not title:
                    title = extract_title_from_first_page(pages, pdf_path)
                if not abstract:
                    abstract = extract_abstract_from_text(pages)

                full_text = "\n".join(pages)
                if not keywords:
                    keywords = extract_top10_keywords(full_text, stop_words)

                sents = nltk.sent_tokenize(full_text)
                evidence = " || ".join(find_top_3_sentences(claim, sents, tokenizer, model, device))

        out_row = [
            index_, year, author, reason, pdf_file, claim, label,
            title, abstract, keywords, evidence
        ]

        pd.DataFrame([out_row], columns=out_cols).to_csv(args.output_csv, mode="a", header=False, index=False)

    print(f"✅ Evidence written to: {args.output_csv}")


if __name__ == "__main__":
    main()
