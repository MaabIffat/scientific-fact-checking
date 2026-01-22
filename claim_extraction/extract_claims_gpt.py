import os
import re
import csv
import time
import argparse
from typing import Dict, Optional, List, Tuple

import fitz  # PyMuPDF
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


# ----------------------------
# Metadata loader
# ----------------------------
def load_metadata_maps(metadata_csv: Optional[str]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    if not metadata_csv:
        return {}, {}, {}

    df = pd.read_csv(metadata_csv)

    if "pdf_file_name" not in df.columns or "title" not in df.columns:
        raise ValueError("metadata_csv must contain columns: pdf_file_name, title (and optionally author, year)")

    key_series = df["pdf_file_name"].astype(str).str.strip().str.lower()

    title_map = dict(zip(key_series, df["title"].astype(str).fillna("Unknown Title")))

    author_map: Dict[str, str] = {}
    if "author" in df.columns:
        author_map = dict(zip(key_series, df["author"].astype(str).fillna("Not found")))

    year_map: Dict[str, str] = {}
    if "year" in df.columns:
        year_map = dict(zip(key_series, df["year"].astype(str).fillna("")))

    return title_map, author_map, year_map


# ----------------------------
# PDF -> text
# ----------------------------
def extract_text_from_pdf(pdf_path: str, max_pages: int = 5) -> str:
    doc = fitz.open(pdf_path)
    parts = []
    for page_num in range(min(max_pages, doc.page_count)):
        parts.append(doc.load_page(page_num).get_text())
    doc.close()
    return "\n".join(parts).strip()


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


# ----------------------------
# Prompt (paper-aligned + parser-aligned)
# ----------------------------
SYSTEM_PROMPT = (
    "You are an advanced scientific research question extractor specializing in automated fact-checking. "
    "Your task is to analyze manuscripts and extract scientific research claims or research questions."
)


def build_user_prompt(title: str, article_text: str, num_claims: int, num_reversed: int) -> str:
    return f"""
Input:
Given the following article [ARTICLE]
Title: {title}

Instructions:
Please perform two tasks.

TASK 1: Claims and research questions
- Identify at least {num_claims} explicitly stated or implied research claims or research questions.
- Claims must be detailed, comprehensive, and complex.
- Do NOT refer to "the authors", "the method", or "the research" generically.
- Instead, name specific techniques, datasets, frameworks, or concepts mentioned in the text.
- For EACH claim, also write a standalone research question that is understandable without context.
  Replace abbreviations/acronyms/pronouns with full forms.

IMPORTANT FORMAT REQUIREMENT:
For EACH item, output EXACTLY this 4-line template (repeat it for every item):

**Claim:** ...
**Research Question:** ...
**Support or Refute Label:** Support OR Refute
**Reason:** ... (4–5 sentences grounded ONLY in the provided article text)

Reversed claims:
- Generate at least {num_reversed} reversed claims.
- Each reversed claim must logically contradict an existing claim.
- Label them with "Refute".
- Avoid explicit negation keywords such as "not" (use antonyms or opposite conditions instead).
- Reversed claims must ALSO use the exact same 4-line template above.
- Do NOT use headings like "Reversed Claim:"; always use "**Claim:**".

TASK 2: Keywords
- Extract 10–20 relevant keywords present in the text.

Provide keywords using EXACTLY:
**Keywords Extracted:** keyword1, keyword2, keyword3, ...

[ARTICLE]
{article_text}
""".strip()


def call_openai(client: OpenAI, model: str, system_prompt: str, user_prompt: str,
                temperature: float, max_output_tokens: int) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    return completion.choices[0].message.content or ""


# ----------------------------
# Keyword extraction (ROBUST)
# ----------------------------
def extract_keywords(response_text: str) -> str:
    """
    Robustly extract keywords from various common formats:
    - **Keywords Extracted:** a, b, c
    - **Keywords Extracted:**\n a, b, c
    - Keywords Extracted: a, b, c
    - **Keywords:** a, b, c
    - Bullets under the heading
    """
    text = response_text.strip()

    # 1) Match the heading line, capture everything after it (including newlines) until end
    patterns = [
        r"\*\*Keywords Extracted:\*\*\s*([\s\S]+)$",
        r"Keywords Extracted:\s*([\s\S]+)$",
        r"\*\*Keywords:\*\*\s*([\s\S]+)$",
        r"Keywords:\s*([\s\S]+)$",
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            chunk = m.group(1).strip()

            # Stop if the model continues with other sections (rare, but safe)
            # (We cut at next **Heading:** if present)
            chunk = re.split(r"\n\s*\*\*[^*]+:\*\*", chunk)[0].strip()

            # Convert bullet lists to comma-separated
            lines = [ln.strip("•-* \t") for ln in chunk.splitlines() if ln.strip()]
            # If it's already one line with commas, keep it
            if len(lines) == 1:
                return lines[0].strip().rstrip(".")

            # If multiple lines, join
            joined = ", ".join(lines)
            return joined.strip().rstrip(".")

    return ""


# ----------------------------
# Parsing
# ----------------------------
def parse_response(response_text: str) -> Tuple[List[Tuple[str, str, str, str]], str]:
    keywords = extract_keywords(response_text)

    pattern = re.compile(
        r"\*\*(Claim|Reversed Claim):\*\*\s*(.+?)\s*"
        r"\*\*Research Question:\*\*\s*(.+?)\s*"
        r"\*\*Support or Refute Label:\*\*\s*(Support|Refute)\s*"
        r"\*\*Reason:\*\*\s*(.+?)(?=\n\s*\*\*(Claim|Reversed Claim):\*\*|\n\s*(?:\*\*Keywords|\bKeywords)\b|$)",
        re.DOTALL | re.IGNORECASE
    )

    rows: List[Tuple[str, str, str, str]] = []
    for _, claim, rq, label, reason, _ in pattern.findall(response_text):
        reason = re.sub(r"\*\*Claim\s*\d+\s*:\*\*", "", reason, flags=re.IGNORECASE)
        rows.append((
            " ".join(claim.split()),
            " ".join(rq.split()),
            label.strip().title(),
            " ".join(reason.split()),
        ))

    return rows, keywords


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract claims/RQs from PDFs using GPT and save to CSV (with title+author+year).")
    parser.add_argument("--input_dir", required=True, help="Folder containing PDF files.")
    parser.add_argument("--output_csv", required=True, help="Output CSV file path.")
    parser.add_argument("--metadata_csv", default=None, help="Metadata CSV with pdf_file_name,title,author,year.")
    parser.add_argument("--model", default="gpt-4", help="Model name (e.g., gpt-4).")
    parser.add_argument("--max_pages", type=int, default=5, help="Read first N pages from each PDF.")
    parser.add_argument("--max_words_input", type=int, default=2500, help="Max words sent to the model.")
    parser.add_argument("--num_claims", type=int, default=12, help="How many claims to request.")
    parser.add_argument("--num_reversed", type=int, default=6, help="How many reversed claims to request.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature.")
    parser.add_argument("--max_output_tokens", type=int, default=2500, help="Max tokens in model output.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between API calls.")
    parser.add_argument("--max_pdfs", type=int, default=0, help="If >0, process only first N PDFs (for testing).")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing. Put it in your environment or in a .env file.")

    title_map, author_map, year_map = load_metadata_maps(args.metadata_csv)

    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    pdf_files = sorted(set(
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(".pdf")
    ))

    if args.max_pdfs and args.max_pdfs > 0:
        pdf_files = pdf_files[:args.max_pdfs]

    if not pdf_files:
        raise SystemExit(f"No PDFs found in: {args.input_dir}")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    client = OpenAI()

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pdf_file_name", "title", "author", "year", "claim", "research_question", "label", "reason", "keywords"])

        for pdf_name in tqdm(pdf_files, desc="Claim extraction"):
            pdf_path = os.path.join(args.input_dir, pdf_name)
            key = pdf_name.strip().lower()

            title = title_map.get(key, "Unknown Title")
            author = author_map.get(key, "Not found")
            year = year_map.get(key, "")

            try:
                pdf_text = extract_text_from_pdf(pdf_path, max_pages=args.max_pages)
                if not pdf_text.strip():
                    writer.writerow([pdf_name, title, author, year, "", "", "", "Empty extracted text", ""])
                    continue

                text_for_model = truncate_words(pdf_text, args.max_words_input)
                user_prompt = build_user_prompt(title, text_for_model, args.num_claims, args.num_reversed)

                response = call_openai(
                    client=client,
                    model=args.model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )

                rows, keywords = parse_response(response)

                if not rows:
                    writer.writerow([pdf_name, title, author, year, "", "", "", "No claims parsed (format mismatch)", keywords])
                else:
                    for claim, rq, label, reason in rows:
                        writer.writerow([pdf_name, title, author, year, claim, rq, label, reason, keywords])

            except Exception as e:
                writer.writerow([pdf_name, title, author, year, "", "", "", f"Error: {e}", ""])

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"✅ Done. Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
