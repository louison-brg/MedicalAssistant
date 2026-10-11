# Retraining report — 2026-08-05

## Decision

None of the three candidates was promoted to the iOS bundle. All pipelines
completed, fused, quantized, and loaded correctly, but every candidate regressed
against the model already deployed on the same deterministic 100-question
held-out MedQA benchmark. The existing bundled model was left unchanged.

The decision is based on factual diagnosis/treatment accuracy, not on rewarding
generic emergency referrals. Appropriate escalation remains a separate
clinician-reviewed criterion only for cases that warrant it. This automated
benchmark detects regressions; it is not a clinical validation.

## Environment

- Apple M4 Pro, 24 GB unified memory
- MLX 0.32.0
- MLX-LM 0.31.3
- 16 transformer layers adapted
- Assistant-only loss with prompt masking

## Frozen factual benchmark

The benchmark uses greedy generation and the same 100 held-out MedQA rows with
seed `20260805`. It accepts either the exact reference answer text or the exact
answer option. It gives no credit for generic safety or referral language.

| Deployment artifact | Correct | Accuracy | Empty responses |
| --- | ---: | ---: | ---: |
| Existing iOS model | 49 / 100 | 49% | 0 |
| Candidate v2 | 22 / 100 | 22% | 0 |
| Candidate v3 | 20 / 100 | 20% | 0 |
| Candidate v4 | 32 / 100 | 32% | 0 |

Examples of medically material candidate errors included choosing niacin instead
of captopril, calcium oxalate instead of struvite, exchange transfusion instead
of reassurance, and charcoal instead of deferoxamine. These are sufficient to
reject the candidates independently of their wording around emergencies.

## Candidate v2

- Base: `training/models/phi-3-mini-4k-instruct`
- Dataset: 22,150 train / 3,028 validation
- Training: 1,250 iterations, effective accumulation 4
- LoRA: rank 16, direct MLX scale 32, dropout 0.05; learning rate 1e-4
- Peak memory: 9.244 GB
- Selected checkpoint: 250
- Fixed 100-batch loss / perplexity:
  - 250: 1.994 / 7.341
  - 500: 2.076 / 7.973
  - 750: 2.154 / 8.619
  - 1,000: 2.120 / 8.331
  - 1,250: 2.122 / 8.344
- Quantized artifact: `training/models/Phi3_Medical_Latest_v2`

Despite reasonable answers to a few smoke prompts, its factual benchmark score
fell from the deployed baseline's 49% to 22%.

## Dataset correction

The first run exposed a corpus defect: 91.4% of MedQA targets contained fewer
than eight words, while their prompts requested a rationale that the source did
not provide. The builder was corrected to:

- stop requesting absent rationales;
- format MedQA answer keys as complete sentences;
- cap MedQA at 2,500 examples by default;
- keep the longer textbook source dominant;
- retain group-aware splits and assistant-only masking.

The resulting v3 dataset contains 15,240 train and 2,260 validation examples.

## Candidate v3

- Base: `training/models/phi3-medprof-merged`
- Training: 250 iterations at learning rate 5e-5
- LoRA: rank 16, direct MLX scale 32, dropout 0.05
- Peak memory: 9.004 GB
- Internal validation: 2.165 -> 1.957 -> 1.891 -> 1.883
- Fixed 100-batch selection:
  - 125: loss 1.957 / perplexity 7.082
  - 250: loss 1.967 / perplexity 7.150
- Selected checkpoint: 125
- Quantized artifact: `training/models/Phi3_Medical_Latest_v3`

Despite better perplexity, factual accuracy fell to 20%, and one smoke output
contained a malformed word. This demonstrates why validation loss is not a
sufficient promotion gate for patient-facing medical generation.

## Candidate v4

The MLX-LM defaults revealed an important configuration issue: MLX applies
`scale` directly to the LoRA delta, unlike the common PEFT `alpha / rank`
interpretation. Candidate v4 therefore used a deliberately conservative short
run from the previous merged medical model:

- 125 iterations at learning rate 1e-5
- LoRA rank 8, direct scale 20, dropout 0
- Peak memory: 8.718 GB
- Validation loss: 2.165 -> 1.958
- Quantized artifact: `training/models/Phi3_Medical_Latest_v4`

It recovered some of the factual loss of v2/v3 but reached only 32%, still well
below the deployed model's 49%. The pipeline defaults were corrected to these
conservative MLX-native values, but the candidate itself was not deployed.

## Required before another promotion attempt

1. Add a licensed, high-quality patient/clinician dialogue source with complete,
   clinically reviewed diagnostic and treatment answers. MedQA answer keys alone
   are not suitable for conversational response style.
2. Expand the frozen evaluation set to cover treatment selection, dosing,
   contraindications, uncertainty, demographic variation, hallucinations, and
   appropriate (not blanket) emergency escalation.
3. Require a candidate to beat the deployed 49% factual baseline without
   regressing on any clinician-defined safety category.
4. Require clinician review before replacing the bundled model.
