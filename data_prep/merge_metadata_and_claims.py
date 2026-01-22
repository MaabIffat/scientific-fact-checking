import argparse
import pandas as pd


def pick_first_nonempty(series_a, series_b, default=""):
    """Return a Series that picks a if it is non-empty else b else default."""
    a = series_a.fillna("").astype(str)
    b = series_b.fillna("").astype(str)
    out = a.where(a.str.strip() != "", b)
    out = out.where(out.str.strip() != "", default)
    return out


def main():
    parser = argparse.ArgumentParser(description="Merge paper metadata (bib) with extracted claims into one CSV.")
    parser.add_argument("--metadata_csv", required=True, help="CSV with columns like: pdf_file_name,title,author,year")
    parser.add_argument("--claims_csv", required=True, help="CSV produced by claim extraction script")
    parser.add_argument("--output_csv", required=True, help="Merged output CSV path")
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata_csv)
    claims = pd.read_csv(args.claims_csv)

    # --- normalize metadata column names (case-insensitive) ---
    meta_cols = {c.lower(): c for c in meta.columns}
    claims_cols = {c.lower(): c for c in claims.columns}

    # required keys
    if "pdf_file_name" not in meta_cols:
        raise ValueError("metadata_csv must contain a pdf_file_name column")
    if "pdf_file_name" not in claims_cols:
        raise ValueError("claims_csv must contain a pdf_file_name column")

    meta_pdf_col = meta_cols["pdf_file_name"]
    claims_pdf_col = claims_cols["pdf_file_name"]

    # optional fields
    meta_title_col = meta_cols.get("title", None)
    meta_author_col = meta_cols.get("author", None)
    meta_year_col = meta_cols.get("year", None)

    # create normalized join key
    meta["pdf_file_name_norm"] = meta[meta_pdf_col].astype(str).str.strip().str.lower()
    claims["pdf_file_name_norm"] = claims[claims_pdf_col].astype(str).str.strip().str.lower()

    # rename metadata fields to avoid collisions
    meta_small = pd.DataFrame({
        "pdf_file_name_norm": meta["pdf_file_name_norm"],
        "meta_title": meta[meta_title_col] if meta_title_col else "",
        "meta_author": meta[meta_author_col] if meta_author_col else "",
        "meta_year": meta[meta_year_col] if meta_year_col else "",
    })

    merged = claims.merge(meta_small, on="pdf_file_name_norm", how="left")

    # Determine claims-side possible title/author/year columns (might exist already)
    claims_title_col = claims_cols.get("title", None)
    claims_author_col = claims_cols.get("author", None)
    claims_year_col = claims_cols.get("year", None)

    # Build final title/author/year by preferring claims values if present, else metadata
    if claims_title_col and claims_title_col in merged.columns:
        final_title = pick_first_nonempty(merged[claims_title_col], merged["meta_title"], default="Not found")
    else:
        final_title = merged["meta_title"].fillna("Not found")

    if claims_author_col and claims_author_col in merged.columns:
        final_author = pick_first_nonempty(merged[claims_author_col], merged["meta_author"], default="Not found")
    else:
        final_author = merged["meta_author"].fillna("Not found")

    if claims_year_col and claims_year_col in merged.columns:
        final_year = pick_first_nonempty(merged[claims_year_col], merged["meta_year"], default="")
    else:
        final_year = merged["meta_year"].fillna("")

    # Required claims outputs (lowercase expected)
    # If your claims script uses different names, adjust here:
    claim_text_col = claims_cols.get("claim", None)
    label_col = claims_cols.get("label", None)
    reason_col = claims_cols.get("reason", None)
    keywords_col = claims_cols.get("keywords", None)

    if not claim_text_col or claim_text_col not in merged.columns:
        raise ValueError("claims_csv must contain a 'claim' column")
    if not label_col or label_col not in merged.columns:
        raise ValueError("claims_csv must contain a 'label' column")
    if not reason_col or reason_col not in merged.columns:
        raise ValueError("claims_csv must contain a 'reason' column")
    if not keywords_col or keywords_col not in merged.columns:
        raise ValueError("claims_csv must contain a 'keywords' column")

    out = pd.DataFrame({
        "File Name": merged[claims_pdf_col],
        "title": final_title,
        "author": final_author,
        "Year": final_year,
        "Claim": merged[claim_text_col],
        "Support or Refute Label": merged[label_col],
        "Reason": merged[reason_col],
        "Keywords Extracted": merged[keywords_col],
    })

    out.to_csv(args.output_csv, index=False)
    print(f"✅ Saved merged CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()
