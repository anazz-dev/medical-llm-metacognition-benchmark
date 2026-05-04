# Data Dictionary

## Trial manifest columns

| Column | Description |
|---|---|
| trial_id | Unique trial identifier |
| vignette_id | Unique vignette identifier |
| evidence_level | Evidence level category |
| information_quality | Clear, conflicting, or missing |
| evidence_score | Internal evidence score used for generation |
| ground_truth_label | AT_NCD, DRCI, or underdetermined |
| forced_choice_accuracy_applicable | Whether forced-choice accuracy is defined |
| expected_more_information_needed | Expected information-sufficiency response |
| prompt_id | Prompt variant identifier |
| option_A | Text of option A |
| option_B | Text of option B |
| vignette_text | Synthetic clinical vignette |
| prompt_text | Full prompt sent to the model |

## Parsed output columns

| Column | Description |
|---|---|
| diagnosis | Parsed model diagnosis output |
| confidence | Parsed confidence rating from 0 to 100 |
| more_information_needed | Parsed information-sufficiency judgment |
| chosen_label | Diagnosis label mapped from model output |
| choice_correct | Whether model choice matched intended label |
| parsed_ok | Whether output was parsed successfully |
