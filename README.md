# Medical LLM Metacognition Benchmark

This repository contains the data, prompts, model outputs, analysis files, and manuscript source for the preprint:

**Large Language Models Show Metacognitive Sensitivity in Medical Reasoning**

## Overview

This benchmark tests whether a medical large language model's confidence tracks:

- diagnostic evidence strength
- conflicting evidence
- missing information
- correctness of its diagnostic choice
- whether more information is needed

The benchmark uses synthetic clinical vignettes comparing:

- probable Alzheimer-type neurocognitive disorder (AT-NCD)
- depression-related cognitive impairment (DRCI)

The dataset contains 45 unique synthetic vignettes. Each vignette is presented under 3 prompt variants, yielding 135 trials per model.

## Repository structure

```text
manuscript/       LaTeX manuscript and manuscript figures
data/             Feature bank, vignettes, and trial manifest
outputs/          Raw and parsed model outputs by model
analysis/         Analysis notebook and figure-generation scripts
figures/          Final exported figures
docs/             Protocol, prompt templates, and data dictionary
```

## Reproducing the analysis

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Open the analysis notebook:

```text
analysis/analysis_notebook.ipynb
```

or run scripts in `analysis/`.

## Important note

All clinical vignettes are synthetic. This benchmark is for model evaluation only. It is not a diagnostic tool and does not provide medical advice.
