# Plan v2: SAC × ACON — threat-driven revision

Supersedes `plan.md`. Same goal, same 2-day budget, same hardware (one RTX 2000 Ada, 15.7 GiB).
What changed: every claim in v1 that could quietly break has been checked against the code and
the data, and the experiment matrix has been rearranged so that the *novelty claim* is the thing
the budget protects.

Numbers below are measured on the box, not estimated, unless marked "est".

---

## Part 1 — Threat register

Ranked by what kills the paper, not by what breaks first.

### T1 · The novelty claim rests on a single seed — CRITICAL (scientific)

The paper's one defensible claim is **R2 (gated KL) > R1 (uniform KL)**. v1 tests it with a paired
bootstrap over examples. That measures *example* variance only. Run-to-run seed noise on this setup
is plausibly ±0.5–1.0 F1, which is the same order as the effect being claimed, so a "significant"
bootstrap CI can be pure seed noise. Three seeds per arm is 25 h — the entire budget.

**Mitigation (what makes this survivable):** the comparison is already *paired at the training
level*. `torch.manual_seed(12345)` is fixed, `instruction_trainer.py` does not shuffle (the shuffle
is done once at data-prep time under `random.seed(0)`), and step truncation is deterministic. So
R1 and R2 see **identical examples in identical order with identical initialisation**, and differ
only in the loss term. That is a much stronger design than two independent runs.

Additional cheap guard: evaluate both arms at **2,500 and 5,000 steps**. If the R2−R1 gap has the
same sign at both checkpoints, it is not an artefact of where we stopped. Cost: +0.5 h.

Report the single-seed limitation explicitly. It is a pilot.

### T2 · The gate confounds *which* examples with *how much* KL — CRITICAL (scientific)

Gated KL at λ=1 applies KL to a fraction ḡ of examples. Uniform KL at λ=1 applies it to all of
them. If R2 beats R1, the first reviewer question is: *"your gate is just a smaller effective λ."*
v1 has no defence against this.

**Mitigation:** run R1 at **λ = ḡ** (the gate rate measured in G0), so R1 and R2 carry the same
expected KL mass and differ *only in which examples receive it*. That is the clean ablation, and
it costs nothing extra. The λ=1 literature setting (KV-Distill-style) becomes optional run R1a.

### T3 · The teacher is zero-shot and is *much* worse on average — HIGH (validity) — **MEASURED**

"Full context succeeds where compression fails" assumes the only difference between teacher and
student is context access. It isn't. The student's trained parts (encoder LoRA, `role_tokens`,
`special_tokens`) have adapted to the MRQA prompt format over 20k steps; the teacher is the frozen
decoder reading raw formatted text with no tuning. A low gate rate could mean "compression is
fine", or merely "the untuned teacher is bad at this format".

**Mitigation:** G0 reports the full NLL-margin distribution and a **gate-rate-vs-threshold curve**,
not one number, so the operating point is chosen from evidence instead of committed to blind.

**G0 result (n=2,000 train):** mean gold-answer NLL is **0.75 for the student, 2.42 for the
teacher** — the zero-shot full-context teacher is far worse on average, median margin −1.50 nats.
This is real and it changes the experiment logic (see "G0 outcome" below). It does *not* kill FGD,
because the teacher is better exactly where it matters.

### T4 · Gold-answer NLL is not the reported metric — HIGH (validity)

The gate fires on teacher-vs-student NLL. The paper reports ROUGE-F1/EM from greedy decoding. The
gate can be well-calibrated in NLL space and invisible in F1.

**Mitigation (free, and the best idea in this revision) — this check passed decisively:** `experiment/release_15x/output/
instruction_inference_results.json` already holds **per-example F1 for the released student over
all 67,854 eval examples**. So G0 can run the teacher-vs-student NLL comparison on *eval* examples
and directly ask: **does "teacher wins in NLL" coincide with "the student actually got it wrong"?**
If the gate does not separate F1=0 from F1=1 examples, FGD is gating on noise and we know *before*
spending 8.4 h on R1+R2. This check did not exist in v1 and cost ~10 min.

**G0 result (n=2,000 eval):** the gate fires on **19.2%** of examples the released student got
completely wrong (F1=0) but only **1.5%** of those it got right (F1>0.5) — a 12x concentration on
failures. Mean F1 is **0.180 when gated vs 0.616 when not**; point-biserial correlation −0.287.
The gate selects the student's failures, which is exactly the ACON contrast. FGD has signal.

### T5 · Naive teacher forward blows VRAM — HIGH (engineering)

Measured on the 320k-example train cache:

| | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| context (`input_ids`) | 541.8 | 371 | 1063 | 1211 | 3710 |
| `lm_targets` | 28.7 | 27 | 44 | 64 | 285 |
| answer tokens | 5.4 | 4 | 12 | 25 | 268 |
| teacher seq (ctx+lm−1) | 569.5 | 401 | 1094 | 1237 | 3747 |

HF's `LlamaForCausalLM` casts logits to float32, so calling the teacher the obvious way
materialises `seq × 128256 × 4` bytes: **634 MB at p99, 1.9 GB at max**, every step, on a card
already holding two 1B models plus optimizer state.

**Mitigation:** never call `LlamaForCausalLM` for the teacher. Call `self.decoder.model(...)` to
get hidden states `[1, seq, 2048]` (≈15 MB), slice the answer positions, then apply `lm_head` to
the slice alone. Answer tokens average 5.4 and max 268, so the teacher logit tensor drops to
**137 MB worst case**. KL is only defined on answer tokens anyway, so nothing is lost.

### T6 · The evaluator silently reuses cached generations — MEDIUM (correctness)

`instruction_evaluator.py` skips generation entirely if
`output/instruction_eval_info_list_0.json` exists. A stride-10 pilot eval would therefore poison a
later full confirm eval, silently, producing 6,786 generations scored against 67,854 golds.

**Mitigation:** stride-aware filenames for every artefact.

### T7 · `device_count` vs. real GPU count silently truncates the eval set — MEDIUM (landmine)

`evaluate()` slices with `len(eval_examples) // training_config["device_count"]` (from config)
while `world_size = torch.cuda.device_count()` (from hardware). Both are 1 today, so it is correct
right now — but a copied config with `device_count: 4` would make rank 0 evaluate the first
quarter and everything downstream would score it without complaint.

**Mitigation:** assert the two agree.

### T8 · 1.66 GB result files — MEDIUM (engineering)

`instruction_inference_results.json` writes the full context back out per example: 1.66 GB for a
full eval. Loading two of those for a paired bootstrap is a memory blowup.

**Mitigation:** write a compact `per_example_scores.json` sidecar (index, subset, rouge-f1,
exact_match) at scoring time; the bootstrap reads only sidecars.

### T9 · "5k steps ≈ 97% of training" — MEDIUM, and it resolved *favourably*

v1 justified 5k steps from training loss (0.859 @5k vs 0.841 @20k). Loss deltas do not imply F1
deltas, so that argument is weak. **But reading `util/utils.py:25` settles it a better way:**
`get_wsd_scheduler` is warmup(300) + `ConstantLR(factor=1.0)` — despite the name there is **no
decay phase**. So a 5k-step run is *exactly* the first 5k steps of the 20k schedule, and
truncation is schedule-neutral. Internal pilot comparisons are clean.

Consequence to respect: absolute pilot numbers sit below the released 54.98 ID / 39.25 OOD.
Compare within the pilot only.

### T10 · λ=1 untuned — MEDIUM

CE and KL are on different scales; KL may swamp CE early. Log `ce`, `kl`, `gate_rate` separately
every step. Smoke gate: if `kl/ce > 5` at step 200, drop λ to 0.5 and say so.

### T11 · The gate anneals itself away — **MEASURED, and larger than expected** — HIGH

Minimising KL where the student is weak makes the student stronger there, which switches the gate
off. Self-limiting by design.

**200-step smoke result:** the effect is steep. Gate rate **0.880 in the first quarter → 0.242 in
the last**, mean 0.525, while CE fell 3.32 → 1.56. G0's converged student sat at 0.095. So the
gate rate is a decreasing function of training progress spanning nearly an order of magnitude.

Two consequences:

1. **It invalidated the original T2 mitigation.** R1's λ was to be 0.095 from G0 — but that is the
   rate against the *fully SFT'd* released student, while the pilot runs start from the *pretrain*
   adapter where the student is far weaker. R1 at λ=0.095 would carry much less KL mass than R2,
   reintroducing exactly the confound T2 exists to remove. **Fix:** R1's λ is now taken from R2's
   *realized* mean gate rate, measured from its own training log. The queue order R0 → R2 → R1
   makes this possible, and `run_pilot.sh` re-derives R1's config automatically before training it
   (`make_pilot_configs.py --lambda_from_run r2_fgd`).
2. R2 does not degenerate into R0 — the gate is still firing at 24% after 200 steps, well clear of
   zero. Watch `realized gate rate` in RESULTS.md over the full 5k-step run.

### T12 · TGA leakage — MEDIUM (correctness)

Appending question tokens to each encoder chunk is only safe because `compress()` keeps just
`mem_real_idx` positions from the trimmed KV, and those indices lie inside the chunk region. Assert
it rather than trust it: decoder KV length == anchor count, and `query_ids` decodes to text ending
in `### Answer:\n` with no answer tokens.

### T13 · TGA must not change the compression ratio — LOW

`mem_position_ids` derives from chunk `start_idx`/`end_idx` only, so anchor count is unaffected by
appended query tokens and the ratio stays 15×. Assert the anchor count matches R0's.

### T14 · Unattended execution — LOW

No `tmux`/`screen` on the box. Use `setsid nohup` (`./remote.sh launch`), `set +e` per stage, a
status file, and the existing resume support so a crash costs one stage, not the night.

### T15 · Novelty (carried over from v1, unchanged)

Query-aware soft compression is taken (SeleCom, QGC); full-context-teacher KL distillation is taken
(KV-Distill). The failure gate is the only defensible contribution — which is exactly why T1 and
T2 are existential rather than cosmetic.

**Dismissed after measurement:** eval throughput. Measured 128 ms/example on the released adapter,
so a stride-10 eval is 14.5 min and a full eval is 2.42 h — v1's estimates were right. Peak
inference VRAM 5.06 GiB.

---

## Part 2 — Revised method

Unchanged in substance from v1; stated precisely.

**FGD — Failure-Gated Distillation.** Per SFT example, with `C = len(input_ids)`:

- student: current SAC path, logits `[:, 1:]` (length `Q+A−1`), `model/modeling.py:122-151`
- teacher: frozen decoder over `input_ids ⊕ lm_targets[:-1]`, hidden states sliced at `[C:]`
  (length `Q+A−1`) — **verified to align 1:1 with the student**, then `lm_head` on the slice only (T5)
- both restricted to answer positions (`instruction_target != -100`)
- gate `w = 1[NLL_teacher(gold) < NLL_student(gold) − m]`, detached, margin `m` from G0
- `loss = CE + λ · w · KL(teacher ‖ student)`, log `ce`, `kl`, `gate_rate`

**TGA — Task-Guided Anchors.** Append `query_ids` to each chunk's encoder input with continuing
position ids and bidirectional attention, so anchors attend to the question; only anchor KV is
kept, so the question never reaches the decoder. Reported as a separate query-aware track (memory
is no longer reusable across questions).

`query_ids` needs **no data-prep change and no cache rebuild**: it is derivable at load time as
`lm_targets[:count(instruction_target == -100) + 1]`, which is exactly `question_ids`. (Verified
against `instruction_prepare_data.py:77-90`.) Rebuilding the 1.9 GB cache would have cost ~30 min
for nothing.

---

## Part 2b — G0 outcome and what it changes

G0 ran before any training (~10 min, not the 0.4 h budgeted). Both questions answered:

- **Kill switch: not triggered.** Train gate rate 9.5% at margin 0 (threshold was 5%). The
  gate-rate curve by margin: 0 → 9.5%, 0.25 → 7.7%, 0.5 → 5.6%, 1.0 → 2.5%. **Margin 0 is the
  operating point** — it has the most signal and the failure separation is already excellent there.
- **Validity: passed decisively** (19.2% vs 1.5% gate rate on failures vs successes).

**What it changes.** The teacher is worse than the student on 90.5% of examples (NLL 2.42 vs 0.75).
So uniform KL at λ=1 does not merely dilute the signal — it should *actively hurt*, dragging the
student toward a worse distribution nine times out of ten. Consequences:

1. **R1 at λ=1 is now a straw man.** "Gated beats uniform" would be nearly guaranteed and therefore
   weak evidence. It is still worth running, but as a *demonstration that naive KV-Distill-style
   distillation fails with this teacher* — a result that motivates the gate — not as the claim.
2. **The real bar moves to R2 > R0.** Gated distillation from a teacher that is weak on average
   must beat no distillation at all. That is the honest test, and it is harder.
3. **R1 mass-matched (λ = 0.095) stays the key ablation** for T2: same expected KL mass, different
   choice of examples.

Priority order for the queue is therefore **R0 → R2 → R1(mass-matched) → R3 → R1a(λ=1)**, so that
if the budget runs short the runs that carry the claim are already done.

## Part 3 — Runs

| Run | What | Purpose | Cost |
|---|---|---|---|
| **G0** ✅ | teacher vs released student gold-answer NLL: 2,000 train + 2,000 eval, margin distribution, gate-rate curve, F1 separation | kill switch **and** T3/T4 validity check | **done, 10 min** |
| **R0** | SAC SFT, 5k steps | matched control | 3.2 h |
| **R2** | + FGD, λ=1 | proposed method | 4.2 h est |
| **R1** | + uniform KL, **λ = R2's realized gate rate** (mass-matched; see T11) | the ablation the claim rests on (T2) | 4.2 h est |
| **R3** | + TGA (+FGD if R2 won) | query-aware track | 4.0 h est |
| evals | stride-10 (6,786 ex) × 4 runs, **plus R1/R2 at step 2,500** (T1) | ranking | 1.5 h |

Pilot ≈ **17.5 h**. Confirm (full 67,854-example eval of winner + R0) ≈ **4.9 h**. Total ≈ **22.4 h**
of the 48 h budget.

Optional, in priority order, if the pilot finishes clean:
1. **R1a** — uniform KL at λ=1, the literature KV-Distill setting; now expected to *lose* to R0,
   which is itself the argument for gating (4.2 h + 0.25 h)
2. **Extension** — 20k-step SFT of the winner (~12.8 h) + full eval (2.4 h), comparable to the
   released 15× numbers

### Decision rules

- ~~**G0 kill switch**~~ — passed (9.5% ≥ 5%).
- ~~**G0 validity**~~ — passed (19.2% vs 1.5%).
- **Primary claim (revised after G0):** **R2 − R0** on macro ID F1 and OOD F1, paired bootstrap
  (1,000 resamples, same examples), 95% CI excluding 0, **and** the same sign at both the 2.5k and
  5k checkpoints. Gated distillation must beat no distillation.
- **Supporting ablation:** R2 − R1 (mass-matched λ=0.095) isolates *which* examples get the KL from
  *how much* KL there is.
- **Motivating negative result:** R1a (λ=1) is expected to fall below R0, showing that naive
  full-context KL distillation fails with a teacher this weak.
- Single seed throughout. Stated as a limitation, not hidden.

---

## Part 4 — Files

| File | Change |
|---|---|
| `model/modeling.py` | SFT branch: teacher KL + gate behind `distill_weight` / `distill_gate` / `distill_margin`; `compress()`: optional `encoder_query`; memory-safe teacher logits (T5); `answer_nll()` helper for G0. Defaults leave SAC bit-identical. |
| `sft/instruction_dataloader.py` | derive and yield `query_ids` (no cache rebuild) |
| `sft/instruction_trainer.py` | `max_train_steps`, `extra_checkpoint_steps` (T1) |
| `sft/instruction_evaluator.py` | `--eval_stride`, stride-aware filenames (T6), `device_count` assert (T7), robust `examples_list` path (T8), compact score sidecar (T8) |
| `util/gate_diagnostic.py` | **new** — G0 |
| `util/paired_bootstrap.py` | **new** — reads sidecars, macro-averages like `paper_scores.py` |
| `scripts/run_pilot.sh` | **new** — crash-tolerant detached queue |
| `util/make_results_md.py` | pilot section |

Reuse, don't rewrite: resume support (`util/utils.py`), `env.sh` (gloo, `MPLBACKEND=Agg`),
`patch_config.py`, `util/paper_scores.py` macro scoring, the released 15× adapter.

## Part 5 — Verification gates

1. **Defaults unchanged** — with the new config keys absent, 20 SFT steps from the released adapter
   reproduce the current loss bit-identically.
2. **Teacher alignment** — assert teacher and student logit slices cover the same target tokens.
3. **No TGA leakage** — `query_ids` ends in `### Answer:\n` with no answer tokens; decoder KV
   length == anchor count; anchor count unchanged vs R0 (T12, T13).
4. **Stride sanity** — re-score the released 15× generations at stride 10 without regenerating;
   macro F1 must land within ~1 point of 54.98 / 39.25.
5. **Smoke** — 200-step R2: `gate_rate ∈ (0,1)`, finite `kl`, `kl/ce < 5`, CE decreasing (T10, T11).

---

## Part 6 — Verification results (all gates passed)

| Gate | Result |
|---|---|
| 1 · defaults unchanged | **BIT-IDENTICAL** on 20 SFT examples vs the pre-change code |
| 2 · teacher alignment | gold-answer NLL **1.93 at offset `len(input_ids)`** vs **12.60** one position off |
| 3 · TGA leakage | 200/200 `query_ids` end in `### Answer:\n` with no answer tokens; decoder KV length == anchor count; anchor counts identical with and without the query (ratio still 15×) |
| 4 · stride sanity | stride-10 re-score of the released generations: **ID 55.40 / OOD 38.92** vs published **54.98 / 39.25** (Δ +0.42 / +0.33) |
| 5 · 200-step FGD smoke | gate ∈ (0,1), KL finite (mean 1.03, max 6.80), `kl/ce` 0.40 when gated — well under the λ-reduction threshold of 5, so **λ=1 stands**; CE 3.32 → 1.56 |

Measured costs: peak training VRAM **6.71 GiB** of 15.7; eval **128 ms/example**; teacher forward
adds roughly 30–40% to step time. The gate-closed path builds no KL graph (G0 says that is the
common case), so FGD's overhead is mostly the teacher forward.
