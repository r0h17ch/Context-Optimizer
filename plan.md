Plan: SAC × ACON — 2-day pilot toward a new compression method
Context
You want a paper that combines SAC (latent compression: anchor tokens inside the context are turned into a few KV "memory" vectors a frozen Llama reads) with ACON (text compression driven by contrasting runs where full context succeeds but compression fails, plus task-guided compression), and beats both. Chosen constraints: SAC's MRQA benchmark, local models only, ≤2 days GPU (one RTX 2000 Ada 16 GB), aiming at a main conference.

Honest framing up front. 2 days of one 16 GB GPU cannot produce a main-conference paper: that needs 3 compression ratios, several seeds, extra baselines, and efficiency analysis (≈3–4 more weeks here). What 2 days can do is a decisive pilot: implement the method, test it against a matched SAC control at 15×, and prove (or kill) the idea with a real significance test. Everything below is scoped to that.

Novelty risk found during research. The two obvious combinations are already published:

question-aware soft compression → SeleCom (arXiv 2602.15856), QGC
distilling a compressed model from a full-context teacher with KL → KV-Distill (arXiv 2503.10337)
So the paper cannot rest on either alone. The claim it can make is ACON's specific idea moved into latent space: train only on the contrast where full context wins and compression loses ("failure-gated" distillation). The pilot is designed so that claim stands or falls on one comparison: gated KL (R2) must beat plain KL (R1), not just plain SAC (R0).

Method
FGD — Failure-Gated Distillation (the ACON contrast, in latent space). SAC's decoder is a frozen Llama-3.2-1B, so the same module fed the raw context is a free full-context teacher (no extra model in memory). Per SFT example:

student: current SAC path (compressed KV → answer logits), model/modeling.py:115-151
teacher: self.decoder(input_ids = input_ids ⊕ lm_targets[:-1]) under torch.no_grad(); its logits at positions len(input_ids): align 1:1 with the student's logits[:, 1:] (both predict instruction_target; mask = != -100, answer tokens only)
gate w = 1[NLL_teacher(gold answer) < NLL_student(gold answer)] (detached) — "full context succeeds where compression fails"
loss = CE + λ · w · KL(p_teacher ‖ p_student) over answer tokens, λ = 1.0 (no tuning budget)
log kl and gate_rate in loss_info
TGA — Task-Guided Anchors (ACON's task-conditioned compressor, in latent space). Append the question tokens to each chunk's encoder input (after the chunk, positions continuing), so bidirectional attention lets anchors attend to the question. Only anchor KV is kept (mem_real_idx in compress(), modeling.py:230-243), so question tokens never reach the decoder — no leakage path. Reported as a separate query-aware track, because compressed memory is no longer reusable across questions (reviewers will ask).

Experiments (all at 15×, initialised from the authors' released pretrain adapter)
Skipping pretraining is what makes 2 days possible: every run starts SFT from experiment/release_15x/output/adapter.pt. Measured: SFT examples average 569 tokens vs 962 in pretraining, and the authors' SFT loss is 0.859 at step 5k vs 0.841 at 20k, so a 5k-step SFT captures ~97% of training and is enough to rank variants.

Run	What	Purpose	GPU
gate	teacher vs released SAC student, gold-answer NLL on 2,000 train examples	kill switch for FGD	0.3 h
R0	SAC SFT, 5k steps	matched control	3.2 h
R2	+ FGD	proposed method	~4.2 h
R1	+ uniform KL (w ≡ 1)	KV-Distill-style ablation; R2 > R1 is the novelty claim	~4.2 h
R3	+ TGA (+ FGD if R2 won)	query-aware track	~4 h
evals	every 10th eval example (6,786, all 12 subsets)	ranking	4 × 0.25 h
Decision rules

gate: if the teacher beats the student on < 5% of examples, FGD has no signal → drop R1/R2, run R0 + R3 only.
win: paired bootstrap (1,000 resamples, same examples) on macro ID F1 and OOD F1; a variant wins only if the 95% CI of (variant − R0) excludes 0.
confirm (5 h): full 67,854-example eval of the winner and R0.
extend, only if ≥20 h remain and the win is clear: 20k-step SFT of the winner (~17 h) + full eval, compared with released SAC 15× (54.98 ID / 39.25 OOD F1).
Sequential GPU total ≈ 17 h pilot + 5 h confirm (+ 20 h optional extension).

Files to change
model/modeling.py — CompressLLM.forward SFT branch: optional teacher KL with gate (config keys distill_weight, distill_gate: none|failure); compress(): optional question tokens per chunk (encoder_query: true). Defaults leave SAC unchanged.
sft/instruction_dataloader.py — also yield query_ids: for train, the question prefix of lm_targets (length = count of -100 in instruction_target + 1); for eval, lm_targets. Verify at implementation that it ends with the "\n### Answer:\n" tokens and contains no answer.
sft/instruction_trainer.py — max_train_steps config key (truncate examples to steps × total_batch_size), so samples_num stays 320000 and the existing cache is reused.
sft/instruction_evaluator.py — --eval_stride k (default 1): slice both the generated examples and the gold examples_list identically (:103 and :231-239).
New util/paired_bootstrap.py — per-example rouge-f1 / exact_match from each run's instruction_inference_results.json, grouped by subset, macro-averaged like util/paper_scores.py (reuse its ID_SUBSETS / OOD_SUBSETS).
New scripts/run_pilot.sh — one detached queue: gate → R0 → R2 → R1 → R3, each SFT + stride eval + scoring, so it runs overnight unattended.
util/make_results_md.py — add a "Pilot (5k-step SFT, 1/10 eval)" section.
Reuse, don't rewrite: resume support (util/utils.py save/load_resume_checkpoint) makes every run crash-safe; env.sh (gloo backend, MPLBACKEND=Agg); patch_config.py; util/paper_scores.py macro scoring; the authors' released 15× adapter.

The stopped 15× pretraining run (checkpoint at step 1,000) isn't needed and stays on disk untouched.

Verification
Defaults unchanged: with the new keys absent, 20 SFT steps from the released adapter give bit-identical loss to the current code (SFT was bit-reproducible in the resume test).
Teacher alignment: unit check that teacher and student logits cover the same target tokens (the argmax at the first answer position matches the gold answer's first token on a handful of examples where both are confident).
No leakage in TGA: assert that query_ids decodes to text ending in ### Answer: with no answer tokens; assert decoder KV length equals the anchor count.
Stride eval sanity: re-score the released 15× generations at stride 10 without generating; the macro F1 should sit within ~1 point of the full 54.98 / 39.25.
Smoke: 200-step R2 run shows gate_rate in (0, 1), finite kl, loss decreasing; then launch the queue.
What the 2 days will not give you (needed for a main-conference submission)
5× and 51× ratios; ≥3 seeds; baselines on the same data (500xCompressor, ICAE, DAST branches; KV-Distill; SeleCom/QGC for the query-aware track); latency/memory measurements; failure-category analysis (ACON-style) of what FGD fixes. Estimate: 3–4 more weeks on this GPU.