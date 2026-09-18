# SAC × ACON — combined-method results

_Generated 2026-09-17 22:10 by `util/make_pilot_results.py`._ For the SAC reproduction record see [RESULTS.md](RESULTS.md); for the threat analysis and run design see [`plan-v2.md`](../plan-v2.md).

## Headline

1. **FGD — failure-gated distillation — produces no net gain.** -0.23 [-0.95, +0.47] ID against the matched control; the interval contains zero. The gate itself is both accurate and effective — it concentrates on the student's failures 12× over its successes, and it converts the -7.82 ID **collapse** that ungated distillation causes into a statistical tie. But the full-context teacher is *weaker* than the compressed student on 90.5% of examples, and no gate can turn a weak teacher into a useful one.
2. **TGA — task-guided anchors — works, and it is the entire result.** (TGA alone) +8.76 [+7.75, +9.71] **\*** ID and +6.66 [+4.28, +9.40] **\*** OOD over the control. At 5,000 steps it reaches **ID 60.13 / OOD 45.28**, beating the *published 20,000-step* SAC 15× checkpoint by **+5.19 ID / +6.02 OOD** on a quarter of the training budget.
3. **The two do not combine.** Adding FGD on top of TGA gives -0.10 [-0.80, +0.59] ID and -0.57 [-2.53, +1.56] OOD — nothing, and if anything slightly negative. So FGD is null in *both* configurations tested: alone against the control, and stacked on TGA. The best-performing arm is also the simplest one.

## What the two methods are

**FGD (Failure-Gated Distillation)** — ACON's contrast idea in latent space. SAC's decoder is a frozen Llama-3.2-1B, so feeding it the *raw* context gives a full-context teacher for free. Per example: `loss = CE + λ · w · KL(teacher ‖ student)` over answer tokens, with `w = 1[NLL_teacher < NLL_student]` — distil only where full context succeeds and compression fails.

**TGA (Task-Guided Anchors)** — ACON's task-conditioned compression in latent space. The question is appended to each encoder chunk so bidirectional attention lets anchors attend to it. Only anchor KV is kept, so the question never enters the decoder's memory and the compression ratio is unchanged.

## Results (5,000-step SFT, 1/10 eval = 6,786 examples)

Two reference points, because they answer different questions:

- **Original SAC 15× (20k steps)** — the authors' published checkpoint: **ID F1 54.95, OOD F1 39.26**.
- **R0 (5k steps)** — the *same method* on this pilot's budget. The R0-to-published gap is the cost of training 4× shorter, not a method difference.

Read `vs R0` for the effect of a method change; read `vs original` for whether a 5k-step variant is already competitive with the published 20k model.

| Run | Role | ID F1 | OOD F1 | ΔID vs R0 (95% CI) | ΔOOD vs R0 (95% CI) | ΔID vs orig | ΔOOD vs orig |
|---|---|---|---|---|---|---|---|
| **R0 · SAC SFT** | matched control | **51.37** | **38.62** | (control) | (control) | -3.57 | -0.65 |
| **R2 · + FGD (gated KL, λ=1)** | proposed | **51.15** | **38.47** | -0.23 [-0.95, +0.47] | -0.14 [-2.04, +1.77] | -3.80 | -0.79 |
| **R1 · + uniform KL (λ=ḡ)** | ablation | **51.06** | **38.17** | -0.32 [-1.02, +0.44] | -0.45 [-2.11, +1.44] | -3.89 | -1.09 |
| **R1a · + uniform KL (λ=1)** | literature | **43.56** | **35.84** | -7.82 [-8.77, -6.88] **\*** | -2.78 [-4.82, -0.54] **\*** | -11.39 | -3.43 |
| **R3 · + TGA + FGD** | query-aware | **60.03** | **44.71** | +8.66 [+7.66, +9.60] **\*** | +6.10 [+3.62, +8.96] **\*** | +5.08 | +5.45 |
| **R3b · + TGA alone** | query-aware | **60.13** | **45.28** | +8.76 [+7.75, +9.71] **\*** | +6.66 [+4.28, +9.40] **\*** | +5.19 | +6.02 |
| _Original SAC 15× (20k steps)_ | _published_ | _54.95_ | _39.26_ | _—_ | _—_ | _—_ | _—_ |

**\*** = the 95% CI of a paired bootstrap (1,000 resamples, stratified by subset, same examples for both arms) excludes 0. Every arm is single-seed, so this CI covers *example* variance only — not run-to-run variance. `vs original` carries no CI: it spans different training budgets and is descriptive.

All arms are paired at the training level — same seed (12345), same data order, same initialisation from the released 15× pretrain adapter, same 5,000-step truncation — and differ only in the loss term. `get_wsd_scheduler` is warmup(300) + constant with no decay phase, so a 5k-step run is exactly the first 5k steps of the 20k schedule.

## Checkpoint consistency (2,500 vs 5,000 steps)

With single-seed arms a bootstrap over examples cannot rule out run-to-run noise, so the second guard is whether each arm's sign against R0 survives a different stopping point. An effect that flips sign between checkpoints is not an effect.

| Run | ID F1 @2.5k | ID F1 @5k | Δ vs R0 @2.5k | Δ vs R0 @5k | sign holds? |
|---|---|---|---|---|---|
| R0 · SAC SFT | 49.68 | 51.37 | (control) | (control) | — |
| R2 · + FGD (gated KL, λ=1) | 48.83 | 51.15 | -0.85 | -0.23 | yes |
| R1 · + uniform KL (λ=ḡ) | 48.49 | 51.06 | -1.19 | -0.32 | yes |
| R1a · + uniform KL (λ=1) | 41.63 | 43.56 | -8.05 | -7.82 | yes |
| R3 · + TGA + FGD | 56.89 | 60.03 | +7.22 | +8.66 | yes |
| R3b · + TGA alone | 56.47 | 60.13 | +6.79 | +8.76 | yes |

| Run | OOD F1 @2.5k | OOD F1 @5k | Δ vs R0 @2.5k | Δ vs R0 @5k | sign holds? |
|---|---|---|---|---|---|
| R0 · SAC SFT | 39.05 | 38.62 | (control) | (control) | — |
| R2 · + FGD (gated KL, λ=1) | 36.74 | 38.47 | -2.31 | -0.14 | yes |
| R1 · + uniform KL (λ=ḡ) | 35.75 | 38.17 | -3.30 | -0.45 | yes |
| R1a · + uniform KL (λ=1) | 34.06 | 35.84 | -4.99 | -2.78 | yes |
| R3 · + TGA + FGD | 44.07 | 44.71 | +5.02 | +6.10 | yes |
| R3b · + TGA alone | 42.53 | 45.28 | +3.48 | +6.66 | yes |

## Finding 1 — the gate works; distillation still does not pay

Failure-gated distillation lands -0.23 [-0.95, +0.47] ID and -0.14 [-2.04, +1.77] OOD against the matched control. Both intervals contain zero and both point estimates are negative — **FGD produces no net gain**.

But that null hides the mechanism working. Ungated distillation from the same teacher is *catastrophic*: R1a (uniform KL, λ=1) lands -7.82 [-8.77, -6.88] **\*** ID and -2.78 [-4.82, -0.54] **\*** OOD — a collapse, comfortably significant. The gate removes almost all of that damage.

| Arm | KL applied to | ΔID vs R0 | damage removed |
|---|---|---|---|
| R1a · uniform KL, λ=1 | every example | -7.82 | — |
| R1 · uniform KL, λ=ḡ | every example, matched mass | -0.32 | 96% |
| **R2 · gated KL, λ=1** | **the ~16% where the teacher wins** | **-0.23** | **97%** |

Distillation from this teacher is clearly harmful at full strength, and both ways of reducing its influence rescue almost all of it.

**But read that table carefully — it does not show that *gating* is what helps.** R1 reaches the same place by simply applying less KL uniformly (λ=ḡ), with no gating at all. The comparison that isolates the gate's *selectivity* from its *dose* is R2 vs R1, which holds the KL mass equal and changes only which examples receive it
  — and that comes out +0.09 [-0.62, +0.80] ID, +0.30 [-1.71, +2.31] OOD: indistinguishable.

So the defensible claim is narrow: **the damage from a weak teacher scales with how much KL you apply, and the gate is one of several ways to apply less.** Choosing *which* examples to distil on — the actual ACON-derived contribution — buys nothing measurable over simply turning the weight down. And neither route beats not distilling at all.

The training loss explains why — final-decile cross-entropy:

| Arm | final-decile CE |
|---|---|
| R0 — no distillation | 0.8974 |
| R2 — gated KL | 0.9136 |
| R1 — uniform KL | 0.9586 |
| R1a — uniform KL λ=1 | 1.2661 |

Distillation from this teacher *hurts* the LM objective, and gating only limits the damage rather than turning it into a gain. The ordering `no distillation < gated < uniform` is exactly what a weak teacher predicts.

### Why — the G0 diagnostic, run before any training

Gold-answer NLL over 2000 train examples: **student 0.75, teacher 2.42** (median margin -1.50 nats). The zero-shot full-context teacher is the *worse* model on 90.5% of examples.

Gate rate by margin (nats): `0` → 9.5%, `0.1` → 8.7%, `0.25` → 7.7%, `0.5` → 5.6%, `1` → 2.5%, `2` → 0.4%

**The gate is nonetheless accurate.** On 2000 eval examples it fires on **19.2%** of the examples the released student got completely wrong (F1=0) but only **1.5%** of those it got right — a **12× concentration on failures**. Mean F1 **0.180 when gated** vs **0.616 when not** (point-biserial -0.287).

So the negative result is specific: *the contrast signal is real, but the teacher that produces it is too weak for KL distillation to exploit it.*

### Realized gate rate during R2

Mean **16.1%**. Decile trace: 33% → 16% → 15% → 15% → 15% → 14% → 14% → 14% → 13% → 13%

The gate anneals as the student improves but stabilises well clear of zero, so R2 was not silently degenerating into R0. R1's λ was set to this realized mean so both arms carry the same expected KL mass and differ only in *which* examples receive it.

## Finding 2 — TGA works, and it is the result

Task-guided anchors give +8.66 [+7.66, +9.60] **\*** ID and +6.10 [+3.62, +8.96] **\*** OOD over the matched control, both intervals clear of zero by a wide margin.

**The attribution is measured, not inferred.** R3b runs TGA with the distillation term switched off entirely:

| Arm | ID F1 | OOD F1 | ΔID vs R0 | ΔOOD vs R0 |
|---|---|---|---|---|
| **R3b · TGA alone** | **60.13** | **45.28** | +8.76 [+7.75, +9.71] **\*** | +6.66 [+4.28, +9.40] **\*** |
| R3 · TGA + FGD | 60.03 | 44.71 | +8.66 [+7.66, +9.60] **\*** | +6.10 [+3.62, +8.96] **\*** |

Adding FGD on top of TGA is worth -0.10 [-0.80, +0.59] ID and -0.57 [-2.53, +1.56] OOD — indistinguishable from zero, with both point estimates negative. **All of the gain is TGA.** The simplest arm in the study is also the best-performing one, which is the form the claim should take.

### Caveats that must travel with this number

1. **Not like-for-like with question-agnostic SAC.** Query-conditioned memory cannot be reused across questions — the context must be recompressed per query. This is a different, weaker operating regime, which is why it is reported as a separate query-aware track.
2. **The idea is not new.** Query-aware soft compression is already published (SeleCom, QGC). The contribution here is its instantiation in SAC's anchor mechanism, not the concept.
3. **Single seed.** See the CI caveat above.

### Leakage verification (on the *trained* adapter, at eval time)

A gain this large is also the signature of the question reaching the decoder, so the assertions were re-run against the finished R3 adapter on the eval split rather than trusted from the implementation-time check:

| Check | Result |
|---|---|
| decoder KV length vs anchor count | equal on every example |
| anchor count with vs without the query | identical |
| effective compression ratio | ~15× (14.7–15.2×) |
| `query_ids` content | ends at `### Answer:`, no gold answer |

The decoder receives the question anyway through `lm_targets`; TGA only lets the *compressor* see it. No information is leaked that the model did not already have.

## Setup

| | |
|---|---|
| Hardware | 1× RTX 2000 Ada (16 GB); peak training VRAM 6.71 GiB |
| Base model | Llama-3.2-1B, frozen decoder; LoRA r=128 on encoder q/v |
| Compression | 15×, chunk size 510, mem size 34 |
| Init | the authors' released 15× *pretrain* adapter |
| Budget | 5,000 optimizer steps (batch 16 via accumulation) per arm |
| Eval | greedy, batch 1, stride 10 → 6,786 of 67,854 examples; 128 ms/example |
| Significance | paired bootstrap, 1,000 resamples, stratified by subset |

### Verification gates (all passed before the runs)

| Gate | Result |
|---|---|
| Defaults unchanged | bit-identical loss to the pre-change code on 20 SFT examples |
| Teacher/student alignment | gold-answer NLL 1.93 at offset `len(input_ids)` vs 12.60 one position off |
| TGA leakage | decoder KV == anchor count; ratio unchanged |
| Stride-10 sanity | re-scored released generations: ID 55.40 / OOD 38.92 vs published 54.98 / 39.25 |
| 200-step FGD smoke | gate ∈ (0,1), KL finite, kl/ce 0.40 when gated, CE 3.32 → 1.56 |

## Limitations

- **Single seed per arm.** The bootstrap CIs cover example variance, not training variance. The arms are paired at the training level, which is stronger than independent runs, but ≥3 seeds remain necessary for a submission.
- **One compression ratio (15×).** 5× and 51× are not run.
- **5,000 steps, not 20,000.** Absolute numbers sit below the published checkpoint; R0 is the control that quantifies that gap.
- **OOD is underpowered** — 964 examples at stride 10 give intervals around ±2 F1.
- **No baselines on the same data** (500xCompressor, ICAE, KV-Distill, SeleCom/QGC), and no latency/memory measurements.
