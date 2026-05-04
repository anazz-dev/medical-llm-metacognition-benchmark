# Benchmark Protocol

## Diagnostic contrast

Probable Alzheimer-type neurocognitive disorder (AT-NCD) versus depression-related cognitive impairment (DRCI).

## Design

- 45 synthetic clinical vignettes
- 3 prompt variants per vignette
- 135 trials per model

## Evidence levels

1. Strong DRCI
2. Moderate DRCI
3. Equivocal
4. Moderate AT-NCD
5. Strong AT-NCD

## Information-quality conditions

1. Clear
2. Conflicting
3. Missing

## Model outputs

Each model returns structured JSON:

```json
{
  "diagnosis": "A or B",
  "confidence": 0-100,
  "more_information_needed": true
}
```

## Primary metrics

- forced-choice accuracy
- mean confidence
- Brier score
- expected calibration error
- AUROC2
- information-sufficiency accuracy
