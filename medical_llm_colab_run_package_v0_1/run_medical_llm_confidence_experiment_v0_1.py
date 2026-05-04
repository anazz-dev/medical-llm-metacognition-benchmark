# Medical LLM Confidence Calibration Experiment Runner v0.1
# Converted from the Colab notebook. For notebook use, open run_medical_llm_confidence_experiment_colab_v0_1.ipynb.

# Colab dependency install
%pip install -q openai pandas numpy scikit-learn statsmodels matplotlib tqdm tenacity

import os
import json
import time
import math
import getpass
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sklearn.metrics import roc_auc_score

print("Imports complete.")

MANIFEST_PATH = Path("medical_llm_trial_manifest_seed1to3_v0_1.jsonl")

if not MANIFEST_PATH.exists():
    try:
        from google.colab import files
        print("Manifest not found. Upload medical_llm_trial_manifest_seed1to3_v0_1.jsonl")
        uploaded = files.upload()
        if "medical_llm_trial_manifest_seed1to3_v0_1.jsonl" in uploaded:
            MANIFEST_PATH = Path("medical_llm_trial_manifest_seed1to3_v0_1.jsonl")
        else:
            # use first uploaded jsonl file
            jsonl_files = [name for name in uploaded if name.endswith(".jsonl")]
            if not jsonl_files:
                raise FileNotFoundError("No JSONL manifest uploaded.")
            MANIFEST_PATH = Path(jsonl_files[0])
    except Exception as e:
        raise FileNotFoundError(
            "Manifest file not found. Upload or place medical_llm_trial_manifest_seed1to3_v0_1.jsonl in the notebook directory."
        ) from e

trials = pd.read_json(MANIFEST_PATH, lines=True)
print(f"Loaded {len(trials)} trials from {MANIFEST_PATH}")
print(trials.columns.tolist())
trials.head(3)

# Basic manifest sanity checks
required_cols = [
    "trial_id", "vignette_id", "seed", "diagnostic_pair", "evidence_level",
    "information_quality", "evidence_score", "ground_truth_label",
    "forced_choice_accuracy_applicable", "expected_more_information_needed",
    "prompt_id", "option_A", "option_B", "vignette_text", "prompt_text"
]

missing = [c for c in required_cols if c not in trials.columns]
if missing:
    raise ValueError(f"Manifest missing required columns: {missing}")

print("Manifest sanity checks passed.")
print(trials[["trial_id", "evidence_level", "information_quality", "ground_truth_label", "prompt_id"]].head())

# ---- User settings ----

MODEL = "gpt-4.1-mini"  # change if needed
TEMPERATURE = 0
MAX_OUTPUT_TOKENS = 200

# Cost safety:
N_TRIALS_TO_RUN = 3      # dry run. Set to None for all trials.
SLEEP_BETWEEN_CALLS = 0.2

RESULTS_DIR = Path("medical_llm_results_v0_1")
RESULTS_DIR.mkdir(exist_ok=True)

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OPENAI_API_KEY: ")

print("Model:", MODEL)
print("Dry-run trial count:", N_TRIALS_TO_RUN)
print("Results directory:", RESULTS_DIR)

MEDICAL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {
            "type": "string",
            "enum": ["A", "B"],
            "description": "The selected option: A or B."
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": "Estimated probability that the selected option is correct, from 0 to 100."
        },
        "more_information_needed": {
            "type": "boolean",
            "description": "Whether the vignette is underdetermined enough that further clinical information is needed."
        }
    },
    "required": ["diagnosis", "confidence", "more_information_needed"],
    "additionalProperties": False
}

SYSTEM_INSTRUCTION = (
    "You are participating in a synthetic medical-reasoning research benchmark. "
    "Return exactly one JSON object matching the requested schema. "
    "Do not provide medical advice. Do not explain your reasoning."
)

print(json.dumps(MEDICAL_RESPONSE_SCHEMA, indent=2))

from openai import OpenAI
client = OpenAI()

run_timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
safe_model_name = MODEL.replace("/", "_").replace(":", "_")
RAW_OUTPUTS_PATH = RESULTS_DIR / f"raw_outputs_{safe_model_name}_{run_timestamp}.jsonl"

def row_to_prompt(row: pd.Series) -> str:
    # The manifest already contains the full prompt text.
    return str(row["prompt_text"])

def response_to_jsonable(response):
    # OpenAI SDK objects usually support model_dump().
    try:
        return response.model_dump()
    except Exception:
        try:
            return json.loads(response.model_dump_json())
        except Exception:
            return {"repr": repr(response)}

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception)
)
def call_model(prompt_text: str):
    kwargs = {
        "model": MODEL,
        "input": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt_text},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "medical_llm_confidence_response",
                "schema": MEDICAL_RESPONSE_SCHEMA,
                "strict": True,
            }
        },
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    # Some models/endpoints may not accept temperature. Remove if your endpoint errors.
    if TEMPERATURE is not None:
        kwargs["temperature"] = TEMPERATURE

    response = client.responses.create(**kwargs)
    return response

def append_jsonl(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Select dry-run or full trial set
run_df = trials.copy()
if N_TRIALS_TO_RUN is not None:
    run_df = run_df.head(int(N_TRIALS_TO_RUN)).copy()

print(f"Running {len(run_df)} trials. Output file: {RAW_OUTPUTS_PATH}")

for _, row in tqdm(run_df.iterrows(), total=len(run_df)):
    start = time.time()
    record = {
        "trial_id": row["trial_id"],
        "vignette_id": row["vignette_id"],
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "run_timestamp": run_timestamp,
        "call_started_at": dt.datetime.now().isoformat(),
    }
    try:
        response = call_model(row_to_prompt(row))
        output_text = getattr(response, "output_text", "")
        record.update({
            "status": "ok",
            "response_id": getattr(response, "id", None),
            "output_text": output_text,
            "raw_response": response_to_jsonable(response),
            "latency_seconds": round(time.time() - start, 3),
        })
    except Exception as e:
        record.update({
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "latency_seconds": round(time.time() - start, 3),
        })
    append_jsonl(RAW_OUTPUTS_PATH, record)
    time.sleep(SLEEP_BETWEEN_CALLS)

print("Run complete.")
print("Raw outputs saved to:", RAW_OUTPUTS_PATH)

def load_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)

raw = load_jsonl(RAW_OUTPUTS_PATH)
print(raw.shape)
raw.head()

def parse_output_text(text):
    if text is None or not str(text).strip():
        return {"parsed_ok": False, "diagnosis": None, "confidence": np.nan, "more_information_needed": None, "parse_error": "empty"}
    try:
        obj = json.loads(text)
        diagnosis = obj.get("diagnosis")
        confidence = obj.get("confidence")
        more_info = obj.get("more_information_needed")
        parsed_ok = diagnosis in ["A", "B"] and isinstance(more_info, bool) and confidence is not None
        try:
            confidence = float(confidence)
        except Exception:
            parsed_ok = False
            confidence = np.nan
        if not (0 <= confidence <= 100):
            parsed_ok = False
        return {
            "parsed_ok": bool(parsed_ok),
            "diagnosis": diagnosis,
            "confidence": confidence,
            "more_information_needed": more_info,
            "parse_error": None if parsed_ok else "schema_or_value_error"
        }
    except Exception as e:
        return {"parsed_ok": False, "diagnosis": None, "confidence": np.nan, "more_information_needed": None, "parse_error": str(e)}

parsed_records = []
for _, row in raw.iterrows():
    parsed = parse_output_text(row.get("output_text"))
    parsed["trial_id"] = row["trial_id"]
    parsed_records.append(parsed)

parsed = pd.DataFrame(parsed_records)
results = trials.merge(raw.drop(columns=[c for c in ["raw_response"] if c in raw.columns]), on="trial_id", how="left")
results = results.merge(parsed, on="trial_id", how="left", suffixes=("", "_parsed"))

def label_from_choice(row):
    choice = row.get("diagnosis")
    if choice not in ["A", "B"]:
        return None
    option_text = row["option_A"] if choice == "A" else row["option_B"]
    option_text_lower = str(option_text).lower()
    if "alzheimer" in option_text_lower:
        return "AT_NCD"
    if "depression" in option_text_lower:
        return "DRCI"
    return None

results["chosen_label"] = results.apply(label_from_choice, axis=1)

def compute_choice_correct(row):
    if not bool(row.get("forced_choice_accuracy_applicable")):
        return np.nan
    if row.get("chosen_label") is None:
        return np.nan
    return row["chosen_label"] == row["ground_truth_label"]

results["choice_correct"] = results.apply(compute_choice_correct, axis=1)

def compute_more_info_correct(row):
    if pd.isna(row.get("more_information_needed")):
        return np.nan
    return bool(row["more_information_needed"]) == bool(row["expected_more_information_needed"])

results["more_info_correct"] = results.apply(compute_more_info_correct, axis=1)
results["confidence_prob"] = results["confidence"] / 100.0

PARSED_RESULTS_PATH = RESULTS_DIR / f"parsed_results_{safe_model_name}_{run_timestamp}.csv"
results.to_csv(PARSED_RESULTS_PATH, index=False)
print("Parsed results saved to:", PARSED_RESULTS_PATH)

results[[
    "trial_id", "evidence_level", "information_quality", "prompt_id",
    "ground_truth_label", "diagnosis", "chosen_label", "confidence",
    "more_information_needed", "choice_correct", "more_info_correct", "parsed_ok"
]].head(10)

def expected_calibration_error(y_true, p_pred, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(p_pred)
    y_true = y_true[mask]
    p_pred = p_pred[mask]
    if len(y_true) == 0:
        return np.nan
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        in_bin = (p_pred >= lo) & (p_pred < hi if i < n_bins-1 else p_pred <= hi)
        if in_bin.any():
            acc = y_true[in_bin].mean()
            conf = p_pred[in_bin].mean()
            ece += (in_bin.mean()) * abs(acc - conf)
    return float(ece)

valid_choice = results[
    (results["forced_choice_accuracy_applicable"] == True) &
    (results["parsed_ok"] == True) &
    (results["choice_correct"].notna())
].copy()

metrics = {
    "n_trials_run": int(len(results)),
    "n_valid_choice_trials": int(len(valid_choice)),
    "malformed_or_error_rate": float(1 - results["parsed_ok"].fillna(False).mean()) if len(results) else np.nan,
}

if len(valid_choice) > 0:
    y = valid_choice["choice_correct"].astype(int).values
    p = valid_choice["confidence_prob"].astype(float).values
    metrics["accuracy"] = float(y.mean())
    metrics["mean_confidence"] = float(np.nanmean(p))
    metrics["brier_score"] = float(np.nanmean((p - y) ** 2))
    metrics["ece_10_bins"] = expected_calibration_error(y, p, n_bins=10)
    if len(np.unique(y)) == 2:
        metrics["auroc2_confidence_correctness"] = float(roc_auc_score(y, p))
    else:
        metrics["auroc2_confidence_correctness"] = np.nan

more_info_valid = results[(results["parsed_ok"] == True) & (results["more_info_correct"].notna())]
if len(more_info_valid) > 0:
    metrics["more_information_needed_accuracy"] = float(more_info_valid["more_info_correct"].astype(int).mean())

METRICS_PATH = RESULTS_DIR / f"metrics_{safe_model_name}_{run_timestamp}.json"
with METRICS_PATH.open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

metrics

# Summary tables
if len(results) > 0:
    summary_by_condition = results.groupby(["evidence_level", "information_quality"], dropna=False).agg(
        n=("trial_id", "count"),
        parsed_rate=("parsed_ok", "mean"),
        mean_confidence=("confidence", "mean"),
        choice_accuracy=("choice_correct", "mean"),
        more_info_rate=("more_information_needed", "mean"),
        more_info_accuracy=("more_info_correct", "mean"),
    ).reset_index()
    display(summary_by_condition)

# Confidence by evidence level
evidence_order = ["strong_AT_NCD", "moderate_AT_NCD", "equivocal", "moderate_DRCI", "strong_DRCI"]
plot_df = results[results["parsed_ok"] == True].copy()
plot_df["evidence_level"] = pd.Categorical(plot_df["evidence_level"], categories=evidence_order, ordered=True)

if len(plot_df) > 0:
    conf_summary = plot_df.groupby("evidence_level", observed=True)["confidence"].mean().reindex(evidence_order)
    plt.figure(figsize=(8, 4))
    plt.plot(conf_summary.index.astype(str), conf_summary.values, marker="o")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Mean reported confidence (0–100)")
    plt.xlabel("Evidence level")
    plt.title("Confidence by evidence level")
    plt.tight_layout()
    plt.show()

# Reliability diagram for forced-choice applicable trials
if len(valid_choice) >= 10:
    cal = valid_choice.copy()
    cal["bin"] = pd.cut(cal["confidence_prob"], bins=np.linspace(0,1,11), include_lowest=True)
    rel = cal.groupby("bin", observed=True).agg(
        mean_confidence=("confidence_prob", "mean"),
        empirical_accuracy=("choice_correct", "mean"),
        n=("trial_id", "count")
    ).dropna().reset_index()

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(rel["mean_confidence"], rel["empirical_accuracy"], marker="o")
    plt.xlabel("Mean reported confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability diagram")
    plt.tight_layout()
    plt.show()

    display(rel)
else:
    print("Reliability plot requires more valid forced-choice trials. Run the full 135-trial set.")

print("Raw outputs:", RAW_OUTPUTS_PATH)
print("Parsed results:", PARSED_RESULTS_PATH)
print("Metrics:", METRICS_PATH)

try:
    from google.colab import files
    print("Use files.download(path) if you want to download outputs from Colab.")
except Exception:
    pass

