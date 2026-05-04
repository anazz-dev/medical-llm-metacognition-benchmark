# Medical LLM Confidence Calibration Experiment — Colab Run Package v0.1

This package contains the first runnable experiment package for the AT-NCD vs DRCI pilot.

## Files

- `run_medical_llm_confidence_experiment_colab_v0_1.ipynb`  
  Main Colab notebook. Loads the manifest, runs model calls, parses results, and computes first-pass calibration metrics.

- `run_medical_llm_confidence_experiment_v0_1.py`  
  Script version of the notebook code.

- `medical_llm_trial_manifest_seed1to3_v0_1.jsonl`  
  Primary input file for the notebook. Contains 135 trial rows.

- `medical_llm_trial_manifest_seed1to3_v0_1.csv`  
  CSV version of the trial manifest.

- `medical_llm_feature_bank_v0_1.csv`  
  Feature bank used to generate the vignettes.

- `medical_llm_vignettes_seed1to3_v0_1.csv`  
  The 45 unique synthetic vignettes.

- `medical_llm_experiment_config_v0_1.json`  
  Experiment metadata/configuration.

## How to use in Colab

1. Open the notebook in Google Colab.
2. Upload `medical_llm_trial_manifest_seed1to3_v0_1.jsonl` when prompted.
3. Enter your API key when prompted.
4. Keep `N_TRIALS_TO_RUN = 3` for the first dry run.
5. Inspect the output JSONL/CSV.
6. Set `N_TRIALS_TO_RUN = None` for the full 135-trial pilot.

## Safety and interpretation

This is a synthetic research benchmark. It is not a diagnostic tool and does not provide medical advice.