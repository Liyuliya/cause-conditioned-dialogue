This project investigates whether extracted hypotheses about a speaker’s concern help a model choose a supportive dialogue strategy, and whether direct preference optimization (DPO) with label-derived pairs improves strategy selection compared with continued supervised fine-tuning.

This repository shares selected methods and tests developed during my MSc thesis.

## Research scope

- Prepared ESConv and extracted cause hypotheses from dialogue histories.
- Compared Qwen 2B and 9B action selectors with and without cause hypotheses, using LoRA fine-tuning across seeds.
- Compared label-derived DPO with continued supervised fine-tuning and evaluated strategy selection on held-out ESConv data.
- Examined adherence to support requests and consistency of selected actions in constructed older-adult scenarios.

ESConv accuracy and macro-F1 measure agreement with recorded strategies. The case study measures how selected actions vary with explicit support requests and related dialogue variants. Response quality and outcomes for older adults were outside the evaluation scope.

## Included code

| File | Purpose |
| --- | --- |
| `src/dpo_loss.py` | Completion-only log probabilities and sigmoid DPO loss for chosen and rejected actions. |
| `src/evaluation.py` | Eight-strategy classification metrics and the paired conversation bootstrap used to compare the effect of cause hypotheses across model sizes on the development set. |
| `tests/` | Synthetic checks for completion scoring, classification metrics, and paired resampling. |

The repository contains selected methods and their tests. The full experimental pipeline and data are excluded.

- Prepared the [ESConv dataset](https://github.com/thu-coai/Emotional-Support-Conversation) ([Liu et al., 2021](https://aclanthology.org/2021.acl-long.269/)) and extracted cause hypotheses from dialogue histories.
