import os
import re
import gc
import json
import time
import argparse
from typing import Dict, Any, Optional, List

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login


LLAMA_MODELS = {
    "8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "70b": "meta-llama/Llama-3.3-70B-Instruct",
}

SETTING_NAMES = [
    "claim",
    "claim_title",
    "claim_evidence",
    "claim_abstract",
    "claim_title_abstract_keywords",
    "claim_title_evidence_abstract_keywords",
]

INSTRUCTIONS = r"""
Return ONLY valid JSON in this format:

{
  "Veracity": "True or False",
  "Justification": {
      "Clarity": "...",
      "Relevance": "...",
      "Consistency": "...",
      "Utility": "..."
  }
}

Definitions for Veracity labels:
- True: "The statement is accurate and there’s nothing significant missing."
- False: "The statement is not accurate or makes a ridiculous claim."

Return valid JSON only (no extra text, no code fences).
""".strip()


# -------- column mapping (handles your final CSV schema) --------
def get_field(row: pd.Series, *candidates: str) -> str:
    for c in candidates:
        if c in row and pd.notna(row[c]):
            return str(row[c])
    return ""


def build_input_block(row: pd.Series, setting: str) -> str:
    """
    Supports your CSV columns:
    Index, Year, author, Reason, File Name, Claim, Support or Refute Label,
    title, Abstract, Keywords Extracted, Evidence
    """
    claim = get_field(row, "Claim", "claim")
    title = get_field(row, "title", "Title")
    abstract = get_field(row, "Abstract", "abstract")
    evidence = get_field(row, "Evidence", "evidence", "Top_3_Relevant_Sentences")
    keywords = get_field(row, "Keywords Extracted", "Keywords_Extracted", "keywords")

    if setting == "claim":
        return f'Claim: "{claim}"'

    if setting == "claim_title":
        return f'Claim: "{claim}"\nTitle: "{title}"'

    if setting == "claim_evidence":
        return f'Claim: "{claim}"\nEvidence: "{evidence}"'

    if setting == "claim_abstract":
        return f'Claim: "{claim}"\nAbstract: "{abstract}"'

    if setting == "claim_title_abstract_keywords":
        return (
            f'Claim: "{claim}"\n'
            f'Title: "{title}"\n'
            f'Abstract: "{abstract}"\n'
            f'Keywords_Extracted: "{keywords}"'
        )

    if setting == "claim_title_evidence_abstract_keywords":
        return (
            f'Claim: "{claim}"\n'
            f'Title: "{title}"\n'
            f'Evidence: "{evidence}"\n'
            f'Abstract: "{abstract}"\n'
            f'Keywords_Extracted: "{keywords}"'
        )

    raise ValueError(f"Unknown setting: {setting}")


def build_prompt(row: pd.Series, setting: str) -> str:
    input_block = build_input_block(row, setting)
    return f"""
### Instructions:
{INSTRUCTIONS}

### Input:
{input_block}

### JSON Response:
""".strip()


def robust_json_parse(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()

    fence = re.search(r"```json\s*([\s\S]*?)\s*```", t, re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    try:
        return json.loads(t)
    except Exception:
        pass

    blocks = re.findall(r"\{[\s\S]*\}", t)
    for candidate in reversed(blocks):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and ("Veracity" in obj or "Justification" in obj):
                return obj
        except Exception:
            continue

    return None


def normalize_justification(just: Any) -> str:
    if isinstance(just, dict):
        return " | ".join(f"{k}: {v}" for k, v in just.items())
    if isinstance(just, list):
        return " | ".join(str(x) for x in just)
    return str(just)


def load_llama(model_key: str, offload_folder: Optional[str] = None):
    model_name = LLAMA_MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16)

    # Optional offload for local memory constraints
    if offload_folder:
        kwargs["offload_folder"] = offload_folder
        kwargs["offload_state_dict"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return tokenizer, model, model_name


def main():
    parser = argparse.ArgumentParser(description="Llama inference runner (reads CSV, writes JSON incrementally).")
    parser.add_argument("--model", required=True, choices=["8b", "70b"])
    parser.add_argument("--setting", required=True, choices=SETTING_NAMES)

    parser.add_argument("--input_csv", default="outputs/final_acl_2024_short_claims.csv")
    parser.add_argument("--output_dir", default="outputs/model_runs")

    parser.add_argument("--max_rows", type=int, default=0, help="If >0, process only first N rows.")
    parser.add_argument("--start_row", type=int, default=0)
    parser.add_argument("--gen_tokens", type=int, default=512)
    parser.add_argument("--sleep", type=float, default=0.0)

    parser.add_argument("--offload_folder", default="", help="Optional offload folder for low VRAM setups.")
    parser.add_argument("--hf_token_env", default="HF_TOKEN", help="HF token env var. If missing, assumes already logged in.")
    args = parser.parse_args()

    # HF login (safe)
    hf_token = os.environ.get(args.hf_token_env, "").strip()
    if hf_token:
        login(hf_token)

    if not os.path.exists(args.input_csv):
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)

    if args.start_row > 0:
        df = df.iloc[args.start_row:].reset_index(drop=True)
    if args.max_rows and args.max_rows > 0:
        df = df.iloc[:args.max_rows].reset_index(drop=True)

    os.makedirs(args.output_dir, exist_ok=True)

    run_name = f"llama_{args.model}__{args.setting}"
    output_json_path = os.path.join(args.output_dir, run_name + ".json")
    output_folder = os.path.join(args.output_dir, run_name)
    os.makedirs(output_folder, exist_ok=True)

    # Resume if exists
    output_json: List[Dict[str, Any]] = []
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                output_json = json.load(f)
            print(f"🔄 Resuming {output_json_path} ({len(output_json)} entries)")
        except Exception:
            print("⚠️ Could not parse existing JSON; starting fresh.")
            output_json = []

    processed_keys = set(str(e.get("Row_ID", "")) for e in output_json)

    tokenizer, model, model_name = load_llama(args.model, offload_folder=(args.offload_folder.strip() or None))
    print(f"✅ Loaded model: {model_name}")

    for i, row in df.iterrows():
        # Prefer Index column in your CSV; fallback to row number
        row_id = str(row.get("Index", i))
        if row_id in processed_keys:
            continue

        prompt = build_prompt(row, args.setting)
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokenized = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=tokenized.input_ids,
                    attention_mask=tokenized.attention_mask,
                    max_new_tokens=args.gen_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated_ids = outputs[:, tokenized.input_ids.shape[-1]:]
            gen = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        except Exception as e:
            gen = ""
            print(f"❌ Generation error at row {row_id}: {e}")

        parsed = robust_json_parse(gen)
        veracity = "Not provided"
        justification = "Not provided"

        if parsed:
            if "Veracity" in parsed:
                veracity = str(parsed["Veracity"])
            if "Justification" in parsed:
                justification = normalize_justification(parsed["Justification"])
        else:
            justification = gen.strip() if gen.strip() else "Generation failed"

        entry = {
            "Row_ID": row_id,
            "Model": model_name,
            "Model_Key": args.model,
            "Setting": args.setting,

            # Carry-through fields (aligned to your CSV)
            "Index": get_field(row, "Index", "index"),
            "Year": get_field(row, "Year", "year"),
            "File_Name": get_field(row, "File Name", "pdf_file_name", "pdf_file"),
            "title": get_field(row, "title", "Title"),
            "author": get_field(row, "author", "Author"),
            "Claim": get_field(row, "Claim", "claim"),
            "Support_or_Refute_Label": get_field(row, "Support or Refute Label", "Ground_Label", "label"),
            "Reason": get_field(row, "Reason", "reason"),
            "Evidence": get_field(row, "Evidence", "evidence", "Top_3_Relevant_Sentences"),
            "Abstract": get_field(row, "Abstract", "abstract"),
            "Keywords_Extracted": get_field(row, "Keywords Extracted", "Keywords_Extracted", "keywords"),

            # Model outputs
            "Veracity": veracity,
            "Justification": justification,
        }

        # per-row txt
        txt_path = os.path.join(output_folder, f"{row_id}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(entry, indent=2, ensure_ascii=False))

        output_json.append(entry)
        processed_keys.add(row_id)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)

        print(f"📌 Saved row {row_id} → {txt_path}")
        torch.cuda.empty_cache()
        gc.collect()

        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"\n✅ Done. Saved JSON to {output_json_path}")
    print(f"📁 Per-row outputs in {output_folder}")


if __name__ == "__main__":
    main()
