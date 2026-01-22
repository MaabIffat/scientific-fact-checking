# decontextualization_claims/run_decontextualization.py

import os
import re
import time
import argparse
import pandas as pd
from openai import OpenAI

from decontextualization_claims.prompts import (
    construct_prompt_task1,
    construct_prompt_task2,
)

DEFAULT_SYSTEM = "You are an expert in scientific claim decontextualization and ambiguity analysis."


def call_openai(client: OpenAI, model: str, prompt: str, system: str, max_tokens: int, temperature: float, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"OpenAI error (attempt {attempt+1}/{retries}): {e}")
            time.sleep(2)
    return ""


def extract_subject_and_criteria(output: str):
    subject_match = re.search(r"SUBJECT:\s*(.*)", output)
    criteria_match = re.search(r"DISAMBIGUATION_CRITERIA:\s*(.*)", output)

    subject = subject_match.group(1).strip() if subject_match else ""
    criteria = criteria_match.group(1).strip() if criteria_match else ""
    return subject, criteria


def main():
    parser = argparse.ArgumentParser(description="Decontextualize claims and extract ambiguity criteria (Task1+Task2).")
    parser.add_argument("--input_csv", required=True, help="Input CSV path.")
    parser.add_argument("--output_csv", default="", help="Output CSV path. Default: <input>_with_outputs.csv")
    parser.add_argument("--n", type=int, default=100, help="Number of rows to process (from top).")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--max_tokens_task1", type=int, default=1800, help="Max tokens for Task 1.")
    parser.add_argument("--max_tokens_task2", type=int, default=800, help="Max tokens for Task 2.")
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="System prompt.")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    out_csv = args.output_csv.strip() or args.input_csv.replace(".csv", "_with_outputs.csv")

    df = pd.read_csv(args.input_csv)

    # Ensure required columns exist
    required = ["Claim", "Title", "Abstract", "Ground_Label", "Keywords_Extracted"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in input CSV: {missing}")

    # Pre-allocate output columns
    decontext_claims = [""] * len(df)
    subjects = [""] * len(df)
    disambig_criteria = [""] * len(df)

    client = OpenAI()

    N = min(args.n, len(df))

    try:
        for idx, row in df.head(N).iterrows():
            claim = str(row["Claim"])
            title = str(row["Title"])
            abstract = str(row["Abstract"])
            label = str(row["Ground_Label"])
            keywords = str(row["Keywords_Extracted"])

            # --- Task 1: decontextualization ---
            prompt1 = construct_prompt_task1(claim, title, abstract, label)
            dc = call_openai(
                client=client,
                model=args.model,
                prompt=prompt1,
                system=args.system,
                max_tokens=args.max_tokens_task1,
                temperature=args.temperature,
            )
            decontext_claims[idx] = dc

            # --- Task 2: ambiguity analysis ---
            prompt2 = construct_prompt_task2(dc, claim, label, title, abstract, keywords)
            t2_out = call_openai(
                client=client,
                model=args.model,
                prompt=prompt2,
                system=args.system,
                max_tokens=args.max_tokens_task2,
                temperature=args.temperature,
            )

            subj, crit = extract_subject_and_criteria(t2_out)
            subjects[idx] = subj
            disambig_criteria[idx] = crit

            print(f"[{idx+1}/{N}] Processed claim.")

    except Exception as e:
        print(f"\n‼️ Stopped early due to error: {e}\nSaving partial results…")
    finally:
        df["DECONTEXTUALIZED_CLAIM"] = decontext_claims
        df["SUBJECT"] = subjects
        df["DISAMBIGUATION_CRITERIA"] = disambig_criteria

        df.to_csv(out_csv, index=False)
        print("✅ Saved to", out_csv)


if __name__ == "__main__":
    main()
