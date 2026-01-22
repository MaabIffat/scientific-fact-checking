import os
import re
import argparse
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def fetch_pdf_links(volume_url: str) -> list[str]:
    r = requests.get(volume_url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    pdf_urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".pdf"):
            pdf_urls.add(urljoin(volume_url, href))
    return sorted(pdf_urls)


def safe_filename_from_url(pdf_url: str) -> str:
    name = pdf_url.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def safe_foldername_from_volume_url(volume_url: str) -> str:
    """
    Turn a volume URL into a stable folder name.
    e.g., https://aclanthology.org/volumes/2024.acl-short/ -> 2024.acl-short
          https://aclanthology.org/volumes/P18-2/ -> P18-2
    """
    s = volume_url.rstrip("/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def download_file(url: str, out_path: str, timeout: int = 60) -> None:
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def download_volume(volume_url: str, base_output_dir: str, sleep_s: float, max_pdfs: int):
    volume_name = safe_foldername_from_volume_url(volume_url)
    output_dir = os.path.join(base_output_dir, volume_name)
    os.makedirs(output_dir, exist_ok=True)

    pdf_urls = fetch_pdf_links(volume_url)
    if max_pdfs and max_pdfs > 0:
        pdf_urls = pdf_urls[:max_pdfs]

    if not pdf_urls:
        print(f"⚠️ No PDFs found on: {volume_url}")
        return

    print(f"\n🔎 {volume_url}")
    print(f"   → Found {len(pdf_urls)} PDFs. Saving to: {output_dir}")

    for pdf_url in tqdm(pdf_urls, desc=f"Downloading {volume_name}", leave=False):
        fname = safe_filename_from_url(pdf_url)
        out_path = os.path.join(output_dir, fname)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            continue

        try:
            download_file(pdf_url, out_path)
        except Exception as e:
            print(f"❌ Failed: {pdf_url} ({e})")
            continue

        time.sleep(sleep_s)


def main():
    parser = argparse.ArgumentParser(description="Download PDFs from one or more ACL Anthology volume pages.")
    parser.add_argument(
        "--volume_url",
        action="append",
        required=True,
        help="ACL Anthology volume URL. Repeat this flag for multiple volumes."
    )
    parser.add_argument(
        "--base_output_dir",
        required=True,
        help="Base directory. Each volume will be saved in its own subfolder."
    )
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between downloads.")
    parser.add_argument("--max_pdfs", type=int, default=0, help="If >0, download only first N PDFs per volume.")
    args = parser.parse_args()

    os.makedirs(args.base_output_dir, exist_ok=True)

    for url in args.volume_url:
        download_volume(url, args.base_output_dir, args.sleep, args.max_pdfs)

    print(f"\n✅ Done. All PDFs saved under: {args.base_output_dir}")


if __name__ == "__main__":
    main()
