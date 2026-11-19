# Supportive Action Selection

Selected implementation components from my MSc thesis, *Cause-Conditioned Action Selection with Preference Optimization for Supportive Dialogue: A Case Study in Older-Adult Companionship*.

The thesis asks whether giving a model an extracted hypothesis about the cause of a speaker’s concern helps it choose a supportive dialogue strategy, and whether direct preference optimization (DPO) with label-derived pairs improves strategy selection beyond continued supervised fine-tuning.

## Research scope

- Prepared ESConv and extracted cause hypotheses from dialogue histories.
- Compared Qwen 2B and 9B action selectors with and without cause hypotheses, using LoRA fine-tuning across seeds 42, 43, and 44.
- Compared label-derived DPO with continued supervised fine-tuning and evaluated strategy selection on held-out ESConv data.
- Examined adherence to support requests and consistency of selected actions in constructed older-adult scenarios.

ESConv accuracy and macro-F1 measure agreement with recorded strategies. The case study measures how selected actions vary with explicit support requests and related dialogue variants. Response quality and outcomes for older adults were outside the evaluation scope.

## Included code

| File | Purpose |
| --- | --- |
| `src/dpo_loss.py` | Completion-only log probabilities and sigmoid DPO loss for chosen and rejected actions. |
| `src/evaluation.py` | Eight-strategy classification metrics and the paired conversation bootstrap used to compare the effect of cause hypotheses across model sizes on the development set. |
| `tests/` | Synthetic checks for completion scoring, classification metrics, and paired resampling. |

The repository contains selected methods and their tests. The full experimental pipeline and data are outside its scope.
