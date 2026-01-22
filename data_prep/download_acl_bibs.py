import os
import re
import argparse
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def fetch_pdf_urls(volume_url: str) -> list[str]:
    """Find all .pdf links on an ACL Anthology volume page."""
    r = requests.get(volume_url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    pdf_urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".pdf"):
            pdf_urls.add(urljoin(volume_url, href))

    return sorted(pdf_urls)


def pdf_url_to_bib_url(pdf_url: str) -> str:
    """
    Convert:
      https://aclanthology.org/2023.acl-short.85.pdf
    to:
      https://aclanthology.org/2023.acl-short.85.bib
    """
    if not pdf_url.lower().endswith(".pdf"):
        raise ValueError(f"Not a PDF url: {pdf_url}")
    return pdf_url[:-4] + ".bib"


def safe_foldername_from_volume_url(volume_url: str) -> str:
    s = volume_url.rstrip("/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def safe_filename_from_url(url: str) -> str:
    name = url.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def download_text(url: str, timeout: int = 60) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def download_volume_bibs(volume_url: str, base_output_dir: str, sleep_s: float, max_items: int):
    volume_name = safe_foldername_from_volume_url(volume_url)
    output_dir = os.path.join(base_output_dir, volume_name)
    os.makedirs(output_dir, exist_ok=True)

    pdf_urls = fetch_pdf_urls(volume_url)
    if max_items and max_items > 0:
        pdf_urls = pdf_urls[:max_items]

    if not pdf_urls:
        print(f"⚠️ No PDFs found on: {volume_url}")
        return

    bib_urls = [pdf_url_to_bib_url(u) for u in pdf_urls]
    print(f"\n🔎 {volume_url}")
    print(f"   → Found {len(bib_urls)} .bib links. Saving to: {output_dir}")

    for bib_url in tqdm(bib_urls, desc=f"Downloading bibs {volume_name}", leave=False):
        fname = safe_filename_from_url(bib_url)  # e.g., 2023.acl-short.85.bib
        out_path = os.path.join(output_dir, fname)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            continue

        try:
            bib_text = download_text(bib_url)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(bib_text)
        except Exception as e:
            # Some entries (front-matter) might not have .bib; we just skip with warning
            print(f"❌ Failed: {bib_url} ({e})")
            continue

        time.sleep(sleep_s)


def main():
    parser = argparse.ArgumentParser(description="Download .bib files from one or more ACL Anthology volume pages.")
    parser.add_argument(
        "--volume_url",
        action="append",
        required=True,
        help="ACL Anthology volume URL. Repeat this flag for multiple volumes."
    )
    parser.add_argument(
        "--base_output_dir",
        required=True,
        help="Base output dir. Each volume will be saved in its own subfolder."
    )
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between downloads.")
    parser.add_argument("--max_items", type=int, default=0, help="If >0, download only first N bibs per volume.")
    args = parser.parse_args()

    os.makedirs(args.base_output_dir, exist_ok=True)

    for url in args.volume_url:
        download_volume_bibs(url, args.base_output_dir, args.sleep, args.max_items)

    print(f"\n✅ Done. All .bib files saved under: {args.base_output_dir}")


if __name__ == "__main__":
    main()
