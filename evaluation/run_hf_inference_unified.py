import os
import re
import gc
import json
import time
import argparse
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login


# -----------------------------
# Model registries
# -----------------------------
LLAMA_MODELS = {
    "8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "70b": "meta-llama/Llama-3.3-70B-Instruct",
}

QWEN_MODELS = {
    "7b": "Qwen/Qwen2-7B-Instruct",
    "72b": "Qwen/Qwen2-72B-Instruct",
}

SETTING_NAMES = [
    "claim",
    "claim_title",
    "claim_evidence",
    "claim_abstract",
    "claim_title_abstract_keywords",
    "claim_title_evidence_abstract_keywords",
]


# -----------------------------
# Paper prompt (as provided)
# -----------------------------
SYSTEM_PROMPT = "You are a helpful assistant for automated fact-checking and temporal analysis of scientific claims."

USER_INSTRUCTIONS = r"""
You are an intelligent decision support system designed for automated fact-checking and temporal analysis (“time-shift”) of scientific claims in the field of Natural Language Processing (NLP). For each claim, assess whether it represents an advanced approach, is outdated, or still holds true even if it is not the most current method. Respond strictly in the following structured JSON format:
{
"Veracity": "True or False",
"Justification": "Detailed reasoning addressing clarity, relevance, consistency, utility",
"Time_Label": "Advanced, Outdated, Still holds true (even if not current)",
"Justification_Time_Label": "Detailed rationale explicitly addressing why this specific temporal label applies. Focus on clarity, relevance, consistency, and utility."
}
Definitions for Veracity labels:
- True: "The statement is accurate and there’s nothing significant missing."
- False: "The statement is not accurate or makes a ridiculous claim."
Definitions for Time_Label:
- Advanced: The claim describes a cutting-edge or state-of-the-art approach that reflects the latest advances in NLP.
- Outdated: The claim refers to a method or model that has been surpassed by newer developments and is no longer commonly used in NLP.
- Still holds true (even if not current): The claim is not the latest, but remains accurate and valid, even though newer alternatives exist in NLP.
Guidelines for Justification:
- Clarity: Concise, coherent, and complete.
- Relevance: Directly relates to claim and context.
- Consistency: Aligns with evidence provided.
- Utility: Useful for evaluating claim accuracy.
Guidelines for Justification_Time_Label:
- A scientific claim related to natural language processing (NLP) is under analysis.
- Clarity: Is this claim now considered old or outdated based on recent advances in NLP?
- Relevance: Are there newer methods, models, or technologies developed that address the same problem or task in a different or improved way? If yes, please briefly describe them.
- Consistency: Matches historical/current evidence. Provide references (preferably from recent years) to papers, tools, or resources that demonstrate the new approaches.
- Utility: If possible, briefly compare the original claim’s approach with the newer developments.
Ensure that all the information is correctly placed in a structured JSON format.
""".strip()


# -----------------------------
# Column mapping (your CSV schema)
# -----------------------------
def get_field(row: pd.Series, *candidates: str) -> str:
    for c in candidates:
        if c in row and pd.notna(row[c]):
            return str(row[c])
    return ""


def build_input_block(row: pd.Series, setting: str) -> str:
    claim = get_field(row, "Claim", "claim")
    title = get_field(row, "title", "Title")
    abstract = get_field(row, "Abstract", "abstract")
    evidence = get_field(row, "Evidence", "evidence", "Top_3_Relevant_Sentences")
    keywords = get_field(row, "Keywords Extracted", "Keywords_Extracted", "keywords")

    if setting == "claim":
        return f"Claim: {claim}"

    if setting == "claim_title":
        return f"Claim: {claim}\nTitle: {title}"

    if setting == "claim_evidence":
        return f"Claim: {claim}\nEvidence: {evidence}"

    if setting == "claim_abstract":
        return f"Claim: {claim}\nAbstract: {abstract}"

    if setting == "claim_title_abstract_keywords":
        return (
            f"Claim: {claim}\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}\n"
            f"Keywords: {keywords}"
        )

    if setting == "claim_title_evidence_abstract_keywords":
        return (
            f"Claim: {claim}\n"
            f"Title: {title}\n"
            f"Evidence: {evidence}\n"
            f"Abstract: {abstract}\n"
            f"Keywords: {keywords}"
        )

    raise ValueError(f"Unknown setting: {setting}")


def build_prompt(setting: str, row: pd.Series, family: str) -> List[Dict[str, str]]:
    """
    Uses your paper prompt:
      SYSTEM: fixed system prompt
      USER: instructions + INPUT block

    We keep family difference:
      - qwen: system + user
      - llama: system + user (safe to use system for both)
    """
    input_block = build_input_block(row, setting)

    user_prompt = f"""{USER_INSTRUCTIONS}

INPUT:
{input_block}
""".strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return messages


# -----------------------------
# JSON parsing helpers
# -----------------------------
def robust_json_parse(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()

    # 1) Strip ```json ... ```
    fence = re.search(r"```json\s*([\s\S]*?)\s*```", t, re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 2) Direct parse
    try:
        return json.loads(t)
    except Exception:
        pass

    # 3) Fallback: find JSON-looking blocks and choose one containing Veracity/Time_Label
    blocks = re.findall(r"\{[\s\S]*\}", t)
    for candidate in reversed(blocks):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and ("Veracity" in obj or "Time_Label" in obj):
                return obj
        except Exception:
            continue

    return None


def normalize_text_field(x: Any) -> str:
    """
    Paper prompt expects strings, but models sometimes return dicts/lists.
    Normalize to a single string.
    """
    if x is None:
        return ""
    if isinstance(x, dict):
        return " | ".join(f"{k}: {v}" for k, v in x.items())
    if isinstance(x, list):
        return " | ".join(str(i) for i in x)
    return str(x)


# -----------------------------
# Model loading
# -----------------------------
def resolve_model_name(family: str, model_key: str) -> str:
    if family == "llama":
        if model_key not in LLAMA_MODELS:
            raise ValueError(f"Invalid llama model: {model_key}. Choose from {list(LLAMA_MODELS.keys())}")
        return LLAMA_MODELS[model_key]
    elif family == "qwen":
        if model_key not in QWEN_MODELS:
            raise ValueError(f"Invalid qwen model: {model_key}. Choose from {list(QWEN_MODELS.keys())}")
        return QWEN_MODELS[model_key]
    else:
        raise ValueError("family must be one of: llama, qwen")


def default_dtype_for(family: str, model_key: str):
    # mirror your original choices
    if family == "qwen" and model_key == "72b":
        return torch.float16
    return torch.bfloat16


def default_gen_tokens_for(family: str, model_key: str) -> int:
    # mirror your original choices
    if family == "qwen" and model_key == "72b":
        return 256
    return 512


def load_model_and_tokenizer(model_name: str, dtype, offload_folder: Optional[str]):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    kwargs = dict(device_map="auto", torch_dtype=dtype)

    if offload_folder:
        kwargs["offload_folder"] = offload_folder
        kwargs["offload_state_dict"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return tokenizer, model


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified runner for LLaMA/Qwen with 6 input settings (Veracity + Time_Label).")
    parser.add_argument("--family", required=True, choices=["llama", "qwen"])
    parser.add_argument("--model", required=True, help="Model key: llama={8b,70b}, qwen={7b,72b}")
    parser.add_argument("--setting", required=True, choices=SETTING_NAMES)

    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_json", required=True)

    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--start_row", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--offload_folder", default="", help="Optional offload folder")

    parser.add_argument("--gen_tokens", type=int, default=0, help="Override max_new_tokens")
    parser.add_argument("--dtype", default="", choices=["", "fp16", "bf16"], help="Override dtype")
    parser.add_argument("--hf_token_env", default="HF_TOKEN", help="Env var name that stores HF token")
    args = parser.parse_args()

    # HuggingFace login (safe)
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

    # Output dirs
    output_json_path = args.output_json
    output_folder = output_json_path.replace(".json", "")
    os.makedirs(output_folder, exist_ok=True)

    # Resume
    output_json: List[Dict[str, Any]] = []
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                output_json = json.load(f)
            print(f"🔄 Resuming: already {len(output_json)} entries in {output_json_path}")
        except Exception:
            print("⚠️ Existing JSON unreadable; starting fresh.")
            output_json = []

    processed = set(str(e.get("Row_ID", "")) for e in output_json)

    # Resolve model
    model_name = resolve_model_name(args.family, args.model)

    # dtype
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = default_dtype_for(args.family, args.model)

    # max_new_tokens
    gen_tokens = args.gen_tokens if args.gen_tokens and args.gen_tokens > 0 else default_gen_tokens_for(args.family, args.model)

    tokenizer, model = load_model_and_tokenizer(
        model_name=model_name,
        dtype=dtype,
        offload_folder=(args.offload_folder.strip() or None),
    )
    print(f"✅ Loaded {args.family}/{args.model}: {model_name} | dtype={dtype} | max_new_tokens={gen_tokens}")

    for i, row in df.iterrows():
        row_id = str(row.get("Index", i))
        if row_id in processed:
            continue

        messages = build_prompt(args.setting, row, args.family)
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokenized = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=tokenized.input_ids,
                    attention_mask=tokenized.attention_mask,
                    max_new_tokens=gen_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated_ids = outputs[:, tokenized.input_ids.shape[-1]:]
            gen = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        except Exception as e:
            print(f"❌ Generation error at row {row_id}: {e}")
            gen = ""

        parsed = robust_json_parse(gen)

        # Defaults
        veracity = "Not provided"
        justification = "Not provided"
        time_label = "Not provided"
        justification_time = "Not provided"

        if parsed:
            if "Veracity" in parsed:
                veracity = normalize_text_field(parsed.get("Veracity"))
            if "Justification" in parsed:
                justification = normalize_text_field(parsed.get("Justification"))
            if "Time_Label" in parsed:
                time_label = normalize_text_field(parsed.get("Time_Label"))
            if "Justification_Time_Label" in parsed:
                justification_time = normalize_text_field(parsed.get("Justification_Time_Label"))
        else:
            # fallback: keep raw text as time justification so nothing is lost
            justification_time = gen.strip() if gen.strip() else "Generation failed"

        entry = {
            "Row_ID": row_id,

            # run metadata
            "Family": args.family,
            "Model_Key": args.model,
            "Model": model_name,
            "Setting": args.setting,

            # carry-through fields from your CSV
            "Index": get_field(row, "Index", "index"),
            "Year": get_field(row, "Year", "year"),
            "File_Name": get_field(row, "File Name", "pdf_file_name", "pdf_file"),
            "title": get_field(row, "title", "Title"),
            "author": get_field(row, "author", "Author"),
            "Claim": get_field(row, "Claim", "claim"),
            "Support_or_Refute_Label": get_field(row, "Support or Refute Label", "Ground_Label", "label"),
            "Reason": get_field(row, "Reason", "reason"),
            "Abstract": get_field(row, "Abstract", "abstract"),
            "Keywords_Extracted": get_field(row, "Keywords Extracted", "Keywords_Extracted", "keywords"),
            "Evidence": get_field(row, "Evidence", "evidence", "Top_3_Relevant_Sentences"),

            # model outputs
            "Veracity": veracity,
            "Justification": justification,
            "Time_Label": time_label,
            "Justification_Time_Label": justification_time,
        }

        # Save per-row txt
        txt_path = os.path.join(output_folder, f"{row_id}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(entry, indent=2, ensure_ascii=False))

        # Append and flush JSON
        output_json.append(entry)
        processed.add(row_id)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)

        print(f"📌 Saved row {row_id} → {txt_path}")

        # free memory
        del tokenized
        if "outputs" in locals():
            del outputs
        torch.cuda.empty_cache()
        gc.collect()

        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"\n✅ Finished. Total entries: {len(output_json)}")
    print(f"📁 Outputs: {output_folder}")


if __name__ == "__main__":
    main()
