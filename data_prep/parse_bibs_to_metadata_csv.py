import os
import re
import csv
import argparse
from typing import Optional, List


# -----------------------------
# BibTeX/LaTeX cleaning helpers
# -----------------------------
LATEX_SIMPLE_REPLACEMENTS = {
    r"{\%}": "%",
    r"{\&}": "&",
    r"{\_}": "_",
    r"{\#}": "#",
    r"{\$}": "$",
    r"{\{}": "{",
    r"{\}}": "}",
    r"\%": "%",
    r"\&": "&",
    r"\_": "_",
}

# Basic accent macros -> unicode (covers most ACL cases)
LATEX_ACCENTS = {
    r"\^": {"a": "â", "e": "ê", "i": "î", "o": "ô", "u": "û",
            "A": "Â", "E": "Ê", "I": "Î", "O": "Ô", "U": "Û"},
    r"\"": {"a": "ä", "e": "ë", "i": "ï", "o": "ö", "u": "ü", "y": "ÿ",
            "A": "Ä", "E": "Ë", "I": "Ï", "O": "Ö", "U": "Ü", "Y": "Ÿ"},
    r"\'": {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú", "y": "ý",
            "A": "Á", "E": "É", "I": "Í", "O": "Ó", "U": "Ú", "Y": "Ý"},
    r"\`": {"a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù",
            "A": "À", "E": "È", "I": "Ì", "O": "Ò", "U": "Ù"},
    r"\~": {"a": "ã", "n": "ñ", "o": "õ",
            "A": "Ã", "N": "Ñ", "O": "Õ"},
    r"\c": {"c": "ç", "C": "Ç"},
}


def debibtex(text: str) -> str:
    """
    Convert common BibTeX/LaTeX patterns to readable text.
    Handles:
      - brace-protected caps: {U}ltra{S}parse -> UltraSparse
      - macro caps: {GPT} -> GPT
      - accents: C{\^o}t{\'e} -> Côte
      - specials: {\%} -> %
    """
    if not text:
        return text

    s = text

    # Optional: handle the ACL quirk {'}  (keeps apostrophe)
    s = s.replace("{'}", "'")

    # 1) Replace common escaped/special tokens
    for k, v in LATEX_SIMPLE_REPLACEMENTS.items():
        s = s.replace(k, v)

    # 2) Convert accent patterns like {\^o} or {\'e} or \"o etc.
    for accent_cmd, mapping in LATEX_ACCENTS.items():
        # Matches: {\^o} or {\^ o} etc.
        pat = re.compile(rf"\{{\s*{re.escape(accent_cmd)}\s*\{{?\s*([A-Za-z])\s*\}}?\s*\}}")
        s = pat.sub(lambda m: mapping.get(m.group(1), m.group(1)), s)

        # Matches: \^o or \'e etc.
        pat2 = re.compile(rf"{re.escape(accent_cmd)}\s*\{{?\s*([A-Za-z])\s*\}}?")
        s = pat2.sub(lambda m: mapping.get(m.group(1), m.group(1)), s)

    # 3) Remove brace-protection around single letters: {U}ltra -> Ultra
    s = re.sub(r"\{([A-Za-z])\}", r"\1", s)

    # 4) Remove brace-protection around acronyms/tokens: {GPT} -> GPT
    s = re.sub(r"\{([^{}]+)\}", r"\1", s)

    # 5) Normalize whitespace
    s = s.replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)

    return s


# -----------------------------
# Field extraction
# -----------------------------
def extract_field(bib_text: str, field: str) -> Optional[str]:
    """
    Extract a BibTeX field value robustly:
      title = {...}
      author = {...}
      year = {...}
    Handles multiline.
    """
    pattern = re.compile(
        rf"{field}\s*=\s*(\{{.*?\}}|\".*?\")\s*,?",
        re.IGNORECASE | re.DOTALL
    )
    m = pattern.search(bib_text)
    if not m:
        return None

    raw = m.group(1).strip()

    # strip outer braces or quotes
    if raw.startswith("{") and raw.endswith("}"):
        raw = raw[1:-1]
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]

    raw = raw.replace("\n", " ").strip()
    raw = re.sub(r"\s+", " ", raw)
    return raw


def clean_authors(author_field: Optional[str]) -> str:
    if not author_field:
        return "Not found"

    # BibTeX author separator
    author_field = re.sub(r"\s+and\s+", ", ", author_field, flags=re.IGNORECASE)
    author_field = re.sub(r"\s+", " ", author_field).strip()

    # Debibtex accents/braces
    author_field = debibtex(author_field)

    return author_field if author_field else "Not found"


def extract_year(bib_text: str) -> str:
    """
    Extract year from bib. Returns a 4-digit year string or empty.
    Example: year = "2024" -> 2024
    """
    raw = extract_field(bib_text, "year")
    if not raw:
        return ""
    # Keep digits only (handles {2024} or "2024")
    digits = re.sub(r"[^\d]", "", raw)
    # Usually 4 digits
    return digits[:4] if len(digits) >= 4 else digits


def bib_filename_to_pdf_filename(bib_filename: str) -> str:
    if bib_filename.lower().endswith(".bib"):
        return bib_filename[:-4] + ".pdf"
    return bib_filename + ".pdf"


def iter_bib_files(root_dir: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".bib"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def main():
    parser = argparse.ArgumentParser(description="Parse ACL Anthology .bib files into a clean metadata CSV (title/author/year).")
    parser.add_argument("--bib_root_dir", required=True, help="Root directory containing downloaded .bib files (may have subfolders).")
    parser.add_argument("--output_csv", required=True, help="Output CSV path.")
    args = parser.parse_args()

    bib_paths = iter_bib_files(args.bib_root_dir)
    if not bib_paths:
        raise SystemExit(f"No .bib files found under: {args.bib_root_dir}")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pdf_file_name", "title", "author", "year"])

        for bib_path in bib_paths:
            with open(bib_path, "r", encoding="utf-8", errors="ignore") as bf:
                bib_text = bf.read()

            bib_filename = os.path.basename(bib_path)
            pdf_file_name = bib_filename_to_pdf_filename(bib_filename)

            title_raw = extract_field(bib_text, "title")
            author_raw = extract_field(bib_text, "author")

            title = debibtex(title_raw) if title_raw else "Not found"
            author = clean_authors(author_raw)
            year = extract_year(bib_text)

            writer.writerow([pdf_file_name, title, author, year])

    print(f"✅ Wrote metadata for {len(bib_paths)} bib files to: {args.output_csv}")


if __name__ == "__main__":
    main()
