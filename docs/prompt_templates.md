# Prompt Templates

## P01: AT-NCD first

Options:
A. Probable Alzheimer-type neurocognitive disorder
B. Depression-related cognitive impairment

## P02: DRCI first

Options:
A. Depression-related cognitive impairment
B. Probable Alzheimer-type neurocognitive disorder

## P03: Named options

Options:
- Probable Alzheimer-type neurocognitive disorder
- Depression-related cognitive impairment

## Required output

```json
{
  "diagnosis": "A or B",
  "confidence": number from 0 to 100,
  "more_information_needed": true or false
}
```
