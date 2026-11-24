# FYP research log: ACON on open models

Single source of truth for the project: what was run, every result, every decision, and how to
operate the lab machines. Update it whenever something changes. Last updated **2026-11-04**.

Reports: `reports/progress_report_2026-11-04.html` (published: https://claude.ai/artifact/K3DDgE7gG6XNMroaUFwpGM).
Direction pitch (24 Oct, published only): https://claude.ai/artifact/7PPFEnjANye3RSxhuwfRWr.
Mechanism write-up: `acon-main/experiments/appworld/CASE_STUDY_LEDGER.md`.

---

## 1. Current state (2026-11-04)

- **2026-11-24 CRITICAL: all "success" numbers above are COMPLETION rates, not correctness.**
  `results.json` "success" = `world.task_completed()` = the agent called
  `apis.supervisor.complete_task()` within 30 steps (see `src/productive_agents/env/appworld/env.py`
  `_check_task_completion`; the `world.evaluate()` line is commented out). The official metric needs
  `appworld evaluate <experiment_name> <split>` (repo README), which we never ran until today.
  Ran it for every arm (M2 task states merged into M1's `experiments/outputs` first; evaluation JSONs
  copied to the scratchpad `evals/`). **Official task-goal completion (TGC), test_normal, 168 tasks:**
  none 9.5 (16) | D.7@4096 10.1 (17) | D.7@2048 5.4 (9) | D.7@1024 3.0 (5) | D.7@512 5.4 (9), repeat 5.4 |
  D.5@4096 9.5 (16) | D.5@2048 6.5 (11) | D.5@1024 3.6 (6) | D.5@512 4.2 (7) | D.7@512 14B comp 2.4 (4) |
  D.7@512+ledger 3.0 (5) | +errors ledger 5.4 (9). Every correct task was also declared complete.
  McNemar on correctness: D.7 4096 vs 512 +10/−2 p=0.039; D.5 4096 vs 512 +10/−1 p=0.012;
  D.7@4096 vs none +8/−7 p=1. Dev TGC 3.5–10.5% for all arms; train 1–6%.
  **Meaning:** qwen2.5-14B (Q4) gets only ~10% of AppWorld test tasks right (ACON paper, gpt-4.1: ~56%).
  The threshold effect survives on correctness (4096 > 512, p<0.05 for both guidelines) but on very
  few tasks; "ACON beats no compression" does NOT hold on correctness. The final report (Nov 23) and
  all earlier tables label completion as "solved" — must be corrected. Any paper needs a stronger agent.


- **All experiments are complete**, including the held-out test set (4 arms × 168 tasks).
  Both lab machines are idle; nothing is queued.
- **Main contribution:** ACON's optimized history guideline (paper Prompt D.7) is
  *threshold-brittle*: 70.2% at the paper's 4096 threshold vs 45.2% at 512 on test-normal
  (p = 1.7e-7). Mechanism: the guideline deletes failure information (summaries mention
  failures 32% vs 87% for the unoptimized D.5), so under frequent compression the agent
  forgets what failed and keeps calling nonexistent APIs.
- **The state ledger fix is only partial:** dev 42.9% → 73.2% (p = 0.0015), but test
  45.2% → 52.4% (p = 0.13, n.s.). The dev number was optimistic: the ledger was designed
  while reading train/dev trajectories. **Quote the test number.**
- **2026-11-19: intervention runs started on M1** (`run_logs/queue7.sh`, dev @512, all 57 tasks):
  tags `acon_hist_keepfail_c16k` (D.7 + D.5's keep-failures line, prompt
  `prompt_history_paper_d7_keepfail.jinja`), `acon_hist_ledger_err_c16k` (ledger, errors
  sections only), `acon_hist_ledger_val_c16k` (ledger, values sections only). Ledger sections are
  set by config `state_ledger_sections`; the default renders identically to the earlier ledger
  (checked on 3,581 real states). M2 was unreachable that day. End-sem review: week of 2026-11-23.
  Plan: pick the best fix on dev, then run it on test_normal (168 tasks).
- **Keep-failures result (dev, 56 common tasks):** D.7 + keep-failures line @512 = **58.9%**
  (33/56), vs paper@512 42.9% (+13/−4, p = 0.049; vs repeat +12/−4, p = 0.077), steps 19.1,
  cap 23, 53.5k agent tok/task. Summaries admitting failures rose 32% → 54% (D.5: 87%).
  Partial recovery, consistent with the mechanism (intervention on the mediator moves the outcome).
- **2026-11-19 21:55:** added `keepfail2` (D.7 + keep line + "tool/API log output" removed from
  D.7's delete rule; prompt `prompt_history_paper_d7_keepfail2.jinja`, tag
  `acon_hist_keepfail2_c16k`). `run_logs/watch78.sh` on M1 stops queue 7 after the errors-only arm
  and runs keepfail2 then values-only. M2 was busy that evening: another student (`harshith`) was
  training on its GPU (6.2 GB, 100%), so it was left alone.
- **2026-11-20 dev intervention results (56 common tasks, @512, `analyze_dev.py`, 16 arms):**
  | Arm | Success | vs paper@512 | vs repeat | LLM summary admits failures | Agent tok/task |
  |---|---|---|---|---|---|
  | paper@512 (D.7) | 42.9 | – | – | 32% | 64.5k |
  | keepfail (D.7 + keep line) | 58.9 | +13/−4 p=0.049 | +12/−4 p=0.077 | 54% | 53.5k |
  | keepfail2 (+ delete rule edited) | 53.6 | +10/−4 p=0.18 | +9/−4 p=0.27 | 56% | 59.9k |
  | ledger errors only | **66.1** | **+16/−3 p=0.0044** | +15/−3 p=0.0075 | 43% (+ ledger always lists errors) | 51.6k |
  | ledger values only | 55.4 | +13/−6 p=0.17 | +12/−6 p=0.24 | 35% | 58.6k |
  | full ledger | 73.2 | +22/−5 p=0.0015 | +21/−5 p=0.0025 | 38% | 53.6k |
  Errors-only vs values-only +10/−4 p=0.18; keepfail2 vs keepfail +4/−7 p=0.55 (editing the delete
  rule did not help further; prompt edits plateau at ~55% admission with the 7B compressor).
  Reading: the errors section carries most of the ledger's gain; values alone are weak. Consistent
  with the mechanism, though the errors-vs-values gap itself is not significant on 56 tasks.
- **2026-11-20 11:40: test_normal run of the errors-only ledger started on M1**
  (`run_logs/queue8.sh`, tag `acon_hist_ledger_err_c16k`, 168 tasks, MEMCAP 40 GB, inhibitor +
  notice running). Expected done ≈ Saturday 21 Nov early morning. M2 still busy (harshith,
  `train.py --dataset prostate`), left alone.
- **2026-11-21 test result, errors-only ledger: did NOT generalize.** test_normal, 167 tasks
  common (task `325d6ec_2` crashed with MemoryError even at 40 GB): errors ledger 81/167 = **48.5%**
  vs paper@512 45.5% (+28/−23, p = 0.58), vs full ledger 52.7% (+23/−30, p = 0.41), vs none 61.7%
  (+27/−49, p = 0.015), vs paper@4096 70.1% (+12/−48, p = 3e-6). LLM summary admits failures
  20% (paper@512 on test: 26%). Second dev→test drop (dev 66.1%). **No fix works on test yet.**
  The robust, quotable result remains the threshold brittleness of D.7.
- **2026-11-22 00:03: threshold sweep started on M1** (`run_logs/queue9.sh`, test_normal, D.7
  history only, thresholds 2048 then 1024; tags `acon_hist_paper_t2048_c16k`,
  `acon_hist_paper_t1024_c16k`). Expected done ≈ Monday 23 Nov early morning.
- **2026-11-22 03:57: M2 free (harshith's training ended), queue started on M2**
  (`run_logs/queue_m2a.sh`, test_normal @512): naive D.5 (`acon_hist_c16k`), then D.7 with a 14B
  compressor (`acon_hist_paper_c14b_c16k`, config `ollama_qwen2.5_14b_c16k_history_paperutco.yaml`,
  same Ollama model as the agent so no extra VRAM). Code on M2 verified identical (md5) to local.
  M2 notice is on DISPLAY=:2 (iiitdmk session); idle-delay set to 0 and lock off, **restore
  900 / true afterwards**. Analysis note: in the c14b arm agent and compressor requests both have
  "14b" in the model name, so split them by request content/role, not by model name.
  **Changed 04:10:** the c14b arm moved to M1 (`run_logs/queue10.sh` waits for queue9); M2's
  `run_logs/stop_m2a_after_naive.sh` kills queue_m2a once the naive arm is DONE, so M2 runs only
  naive. Pull c14b results from M1, naive results from M2. Expected: M1 sweep ≈ Sun evening,
  naive on M2 ≈ Sun night, c14b on M1 ≈ Mon morning.
- **2026-11-22 15:00:** threshold 2048 done (test, 167 common tasks, `325d6ec_2` excluded):
  **66.5%** (111/167) vs @512 45.5% (+47/−12, p = 5e-6), vs @4096 70.1% (+17/−23, p = 0.43),
  vs none 61.7% (+41/−33, p = 0.42). Break point lies between 512 and 2048; 1024 running.
  Token numbers: recompute with all request logs (my quick script read only one file per arm).
  Task `325d6ec_2` hung for 6 h in the 1024 arm (100% CPU, earlier MemoryError in the errors-ledger
  arm); killed (rc=143, counts as failure). Added `run_logs/task_timeout.sh` on both machines
  (kills any task > 90 min). **Exclude `325d6ec_2` from all test comparisons** (n = 167).
- **2026-11-22 16:20: naive D.5 @512 on test (M2) — changes the story.** 82/167 = **49.1%**,
  vs D.7@512 45.5% (+31/−25, p = 0.50), vs none 61.7% (+25/−46, p = 0.017), vs D.7@4096 70.1%
  (+12/−47, p = 5e-6). Naive summaries admit failures 85% on test (D.7: 26%), yet it collapses too.
  On dev naive@512 was 69.6% (= none); that did not hold on test. **So on held-out data the
  collapse at 512 is caused by compression frequency, not by D.7's guideline; the "D.7 deletes
  failures" mechanism is a dev-only finding and must not be claimed as the cause.** Third dev→test
  reversal (after full ledger and errors ledger): the 56-task dev split is unreliable for selection.
- **2026-11-22 17:44:** c14b arm moved back to M2 (`run_logs/queue_m2b.sh`, idle since 16:20);
  M1 `queue10.sh` cancelled. M1 runs only the 1024 arm (≈ done 00:30 Mon); c14b on M2 ≈ Mon morning.
- **2026-11-23 FINAL TEST TABLE (test_normal, 167 common tasks, `325d6ec_2` excluded).** All
  experiments done; both machines idle and cleaned (notices removed, M2 idle-delay 900 / lock true).
  | Arm | Solved | Rate | Steps | Cap | Peak | Agent tok/task | All-LLM tok/task | Compr./task | LLM s/task |
  |---|---|---|---|---|---|---|---|---|---|
  | none | 103 | 61.7 | 18.8 | 64 | 8,806 | 116.3k | 116.3k | 0 | 199 |
  | D.7@4096 | 117 | 70.1 | 18.3 | 50 | 6,584 | 80.4k | 84.3k | 0.7 | 181 |
  | D.7@2048 | 111 | 66.5 | 18.6 | 56 | 5,013 | 66.0k | 71.5k | 1.9 | 178 |
  | D.7@1024 | 91 | 54.5 | 21.2 | 76 | 4,237 | 64.4k | 72.8k | 4.5 | 207 |
  | D.7@512 | 76 | 45.5 | 22.7 | 91 | 3,768 | 63.6k | 74.8k | 9.0 | 255 |
  | D.5 (naive)@512 | 82 | 49.1 | 21.6 | 85 | 4,096 | 62.2k | 72.7k | 8.9 | 253 |
  | D.7@512, 14B compressor | 78 | 46.7 | 22.4 | 89 | n/a | n/a | 75.0k | 9.4 | 281 |
  | D.7@512 + ledger | 88 | 52.7 | 21.7 | 79 | 4,173 | 67.1k | 89.0k | 15.6 | 332 |
  | D.7@512 + errors ledger | 81 | 48.5 | 22.7 | 86 | 4,025 | 65.0k | 79.3k | 11.4 | 274 |
  McNemar: 1024 vs 512 +34/−19 p=0.053; 1024 vs 2048 +15/−35 p=0.0066; 1024 vs 4096 p=0.00086;
  1024 vs none +30/−42 p=0.19; c14b vs D.7@512 +25/−23 p=0.89; c14b vs none p=0.0041; c14b vs D.5@512
  p=0.69. Paired bootstrap 95% CI (2,000 resamples, seed 0): 4096−512 +24.6 [+16.2,+32.9]; 4096−none
  +8.4 [+0.0,+17.4]; 512−none −16.2 [−25.7,−6.6]; D.5−D.7 @512 +3.6 [−5.4,+12.0]; c14b−7b +1.2
  [−7.2,+9.6]; ledger−512 +7.2 [−1.8,+16.2]. n=167 McNemar: 4096 vs 512 +53/−12 p=2.8e-7; 512 vs none +21/−48 p=0.0016; ledger vs 512 +33/−21 p=0.13.
  **Findings:** monotone threshold curve; neither guideline, compressor size nor ledger rescues 512;
  compressing more often does not even save tokens (512 uses more all-LLM tokens than 2048, via
  longer trajectories). The ledger lengthens summaries and so triggers more compressions (15.6/task).
  Note: 168-task numbers in §4.6 differ slightly (n=168); use these n=167 numbers in reports.
- **2026-11-23 09:21: extra test runs started (machines were idle after the final report).**
  M1 `run_logs/queue11.sh`: naive D.5 @4096 (`acon_hist_t4096_c16k`) then @2048 (`acon_hist_t2048_c16k`).
  M2 `run_logs/queue_m2c.sh`: naive D.5 @1024 (`acon_hist_t1024_c16k`) then a repeat of D.7 @512
  (`acon_hist_paper_rep2_c16k`, test split) for run-to-run stability. Purpose: full guideline ×
  threshold curve on test (is the guideline irrelevant at every threshold?) and noise check.
  Watchdogs `run_logs/task_timeout_queue11.sh` / `task_timeout_queue_m2c.sh` (90 min per task).
  M2 idle-delay 0 / lock off again — **restore 900 / true afterwards**. Expected done ≈ Tue 24 Nov.
  Report `final_report.tex/.pdf` (repo root) does not include these yet.
- **2026-11-24 GUIDELINE × THRESHOLD GRID ON TEST (all done; machines idle and cleaned, M2 settings restored).**
  165 tasks common to all 10 test arms (excluded: `325d6ec_2`; `325d6ec_3` killed by watchdog in
  D.5@2048; `90adc3f_2` MemoryError in D.5@4096). Success % (cap-hit %, compressions/task, all-LLM tok/task):
  | T_hist | D.7 (ACON optimised) | D.5 (unoptimised) | D.5 − D.7 (McNemar, bootstrap CI) |
  |---|---|---|---|
  | 4096 | 70.3 (29.7, 0.7, 83.9k) | 72.7 (27.3, 0.6, 79.7k) | +15/−11 p=0.56, +2.4 [−3.0,+8.5] |
  | 2048 | 67.3 (32.7, 1.9, 70.9k) | 70.3 (29.7, 1.9, 69.6k) | +21/−16 p=0.51, +3.0 [−4.2,+10.3] |
  | 1024 | 54.5 (45.5, 4.5, 72.8k) | 61.8 (38.2, 4.0, 68.4k) | +29/−17 p=0.10, +7.3 [−0.6,+15.2] |
  | 512  | 46.1 (53.9, 9.0, 74.5k) | 49.1 (50.9, 9.0, 72.7k) | +30/−25 p=0.59, +3.0 [−5.5,+12.1] |
  None 61.8 (38.2). D.7@512 repeat (M2) 46.7, agrees on 164/165 tasks → runs are stable.
  D.5@4096 vs none +40/−22 p=0.03, +10.9 [+1.8,+20.0]; D.5@4096 vs D.5@512 p=2.7e-7; D.5@1024 vs @512 p=0.0046.
  **Findings:** same monotone threshold curve for both guidelines; the optimised guideline is never
  better than the unoptimised one at any threshold (D.5 ≥ D.7 everywhere, n.s.). Compression frequency,
  not the guideline, decides success. Final report (submitted version) predates this grid.
- **Next step (agreed direction):** intervention test of the mechanism on dev, then test:
  (a) errors-only ledger vs values-only ledger, (b) D.7 plus D.5's line "Include key info from
  errors or failed attempts to prevent repeated mistakes". If failure preservation is causal,
  a failure-preserving guideline becomes the improvement.
- Professor's expectation: take a base paper and improve it. Base paper = ACON (ICML 2026).
  No money for APIs; can ask for a better college GPU.

## 2. Papers

| Paper | Role |
|---|---|
| ACON (Kang et al., ICML 2026, arXiv 2510.00615) `ACON_Optimizing_Context_.pdf` | **Base paper**. History + observation compression for agents; guideline optimization (UT/UTCO) with an o3 optimizer; distillation. AppWorld Table 1: none 56.0, Prompting 43.5, UT 51.2, UTCO 56.5 (history); combined Table 14: 44.6 (UTCO). Optimized prompts printed in Appendix D: D.5 history *before* optimization, D.6 UT, D.7 UTCO; D.8 obs before, D.9 UT, D.10 UTCO (PDF pages 30–34). |
| SAC (Liu et al., ICLR 2026, arXiv 2510.08907) `SAC.pdf` | Soft KV compression for single-turn QA. Reproduced; set aside as the main line (needs same model as encoder/decoder, lossy for exact state, agent version published as Anchored Context Distillation 2609.31430). |
| Related, must cite | FOCUS 2609.37590 (test-time, beats ACON on AppWorld 64.9); Beyond Token Savings 2609.32961; Execution Instability 2608.06503 (AppWorld, counterfactual continuations); StateComp 2609.27298; What Does Compression Cost an Agent 2608.16370; The Complexity Trap 2508.21433 (observation masking baseline); TokenPilot 2606.17016; CommitKV 2608.07855; AGORA 2605.26596. |

Literature check result: keeping exact values outside a summary is established practice
(Mem0, vecr-compress, Google ADK session state), so the ledger is **not** a novel idea. No
work found showing ACON-style optimized guidelines are threshold-brittle or delete failures.

## 3. Setup and deviations from the paper

| Item | ACON paper | Ours |
|---|---|---|
| Agent | gpt-4.1 | qwen2.5:14b Q4_K_M via Ollama 0.20.0 (variant `qwen2.5-14b-c16k`, num_ctx 16384) |
| Compressor | gpt-4.1 / distilled Qwen3-14B | qwen2.5:7b (variant `qwen2.5-7b-c16k`) |
| Guidelines | optimized with o3 | **not re-optimized**; D.7 / D.10 copied verbatim into `prompts/context_opt/prompt_history_paper_d7_utco.jinja`, `prompt_obs_paper_d10_utco.jinja` |
| Thresholds | 4096 / 1024 (text); upstream `gpt-4.1_obs.yaml` uses obs 256 | history 512 or 4096, obs 256 |
| Steps, decoding | – | max 30 steps, temperature 0, seed 42 |
| Agent prompt | upstream | upstream + one extra Spotify few-shot example + "output raw Python only" system message (all arms) |
| Splits | train opt., test-normal results | train 90, dev 57 (development), **test_normal 168 held out** (no overlap) |

Model weight hashes (identical on both machines): qwen2.5:14b `sha256-2049f5674b1e…`, qwen2.5:7b `sha256-2bada8a74506…`.

## 4. Results

All numbers regenerate from logs: `analyze_rq1.py` (train+dev, naive arms), `analyze_dev.py`
(12 dev arms), `analyze_cfr.py` (CFR), run from `acon-main/experiments/appworld`.
"naive" = repo-shipped guideline (= paper D.5 = the paper's *Prompting* baseline); "paper" = D.7/D.10.

### 4.1 SAC (MRQA, Llama-3.2-1B)
Reproduction: 15× ID/OOD F1 54.98/39.25 (paper 54.95/39.26); 5× 63.70/47.71 (63.63/47.72).
Pilot (15×, 5k steps, paired): R0 51.37/38.62; R2 FGD −0.23 [−0.95,+0.47] ID; R1 uniform KL −0.32;
R1a KL λ=1 −7.82; R3 TGA+FGD +8.66 [+7.66,+9.60] ID / +6.10 OOD; R3b TGA alone +8.76 / +6.66.
FGD null because the zero-shot teacher was worse than the student on 90.5% of examples.
Details: `SAC/RESULTS_SAC_ACON.md`.

### 4.2 September ACON pilot (superseded)
Ollama default num_ctx 4096 silently truncated 32 of 38 baseline tasks (median final prompt ~10k).
Baseline 12/38 = 31.6%. Not usable.

### 4.3 RQ1 with the naive guideline, 512/256, train+dev 147 tasks
| Arm | Solved | Rate | Train | Dev | Steps | Peak | Agent tok/task | Compr. | LLM s/task |
|---|---|---|---|---|---|---|---|---|---|
| none | 100 | 68.0 | 66.7 | 70.2 | 18.0 | 8,256 | 104.5k | 0 | 178.8 |
| history | 87 | 59.2 | 52.2 | 70.2 | 20.1 | 4,521 | 57.5k | 1,213 | 242.0 |
| observation | 101 | 68.7 | 65.6 | 73.7 | 17.9 | 6,354 | 85.8k | 235 | 188.9 |
| both | 90 | 61.2 | 54.4 | 71.9 | 19.7 | 3,108 | 53.8k | 1,291 | 233.4 |

Interaction I = +1.4, 95% bootstrap CI [−10.2, +12.2] (paper −9.5). McNemar: hist vs none +23/−36
p=0.117; obs +25/−24 p=1; both +27/−37 p=0.26; both vs hist +15/−12 p=0.70.
39-task subset (train_history_tiny) earlier: 64.1/56.4/64.1/59.0, I=+2.6.

### 4.4 Dev, 12 arms (56 tasks common to all)
| Arm | Solved | Rate | Steps | Cap hits | Peak | Agent tok/task | Compr. | LLM s/task |
|---|---|---|---|---|---|---|---|---|
| none | 39 | 69.6 | 18.0 | 17 | 8,494 | 105.4k | 0 | 182.9 |
| hist naive@512 | 39 | 69.6 | 18.4 | 17 | 4,293 | 53.4k | 397 | 214.9 |
| hist naive@4096 | 38 | 67.9 | 17.4 | 18 | 6,459 | 73.5k | 41 | 164.3 |
| hist paper@512 | 24 | 42.9 | 22.2 | 32 | 4,904 | 64.5k | 510 | 282.4 |
| hist paper@512 repeat | 25 | 44.6 | 21.7 | 31 | 4,887 | 61.9k | 465 | 258.0 |
| hist paper@4096 | 41 | 73.2 | 17.1 | 15 | 6,450 | 70.5k | 39 | 168.5 |
| hist paper@512 + ledger | 41 | 73.2 | 17.7 | 15 | 4,468 | 53.6k | 685 | 273.4 |
| obs naive | 41 | 73.2 | 16.5 | 15 | 6,143 | 79.5k | 81 | 206.6 |
| obs paper | 38 | 67.9 | 18.2 | 18 | 6,360 | 85.8k | 115 | 190.3 |
| both naive@512 | 40 | 71.4 | 18.0 | 16 | 3,206 | 50.2k | 442 | 214.4 |
| both paper@512 | 32 | 57.1 | 20.2 | 24 | 3,041 | 55.8k | 497 | 236.0 |
| both paper@4096 | 40 | 71.4 | 18.1 | 16 | 5,339 | 71.2k | 109 | 159.5 |

Key McNemar (dev): paper@512 vs naive@512 +2/−17 p=0.0007; paper@4096 vs paper@512 +20/−3 p=0.0005;
ledger vs paper@512 +22/−5 p=0.0015 (vs repeat p=0.0025); repeat vs original +1/−0 (55/56 same).
Interactions (dev): naive@512 −1.8; paper@512 +16.1; paper@4096 (57 tasks) 0.0.

### 4.5 Mechanism (dev; executed steps from `env_history.json`)
| History arm | Success | Summary admits failures | Steps calling nonexistent APIs | CFR | Token kept | Exact repeats |
|---|---|---|---|---|---|---|
| none | 70 | – | 37% | – | – | 13% |
| naive@512 | 70 | 87% | 36% | 28% | 0% | 13% |
| naive@4096 | 68 | 51% | 37% | 21% | – | 10% |
| paper@512 | 44 | 32% | 46% | 34% | 23% | 13% |
| paper@512 repeat | 45 | 30% | 45% | 35% | 21% | 12% |
| paper@4096 | 74 | 35% | 36% | 29% | 17% | 8% |
| paper@512 + ledger | 74 | 38% (LLM part) | 32% | 61% | 100% | 15% |
| keepfail@512 (56 tasks) | 59 | 54% | – | – | – | – |
| keepfail2@512 (56 tasks) | 54 | 56% | – | – | – | – |
| ledger errors only (56) | 66 | 43% (LLM part) | – | – | – | – |
| ledger values only (56) | 55 | 35% (LLM part) | – | – | – | – |

Ruled out: value loss (CFR does not predict success, r = −0.08), token loss, API-list loss
(all arms keep 1–3% of api_docs names), prose replies (2–5%), exact loops. Supported: D.7 drops
failures (D.5: "Include key info from errors…"; D.7: "Delete tool/API log output"). At 512 there
are ~9 compressions per task; at 4096 fewer than 1. Example task: `d4e9306_2`.

### 4.6 Held-out test (test_normal, 168 tasks, history compression, paper guideline)
| Arm | Solved | Rate | Steps | Cap hits | Peak | Agent tok/task | Compr. | LLM s/task |
|---|---|---|---|---|---|---|---|---|
| none | 104 | 61.9 | 18.8 | 64 | 8,778 | 115.8k | 0 | 197.9 |
| paper@4096 | 118 | 70.2 | 18.3 | 50 | 6,584 | 80.4k | 124 | 182.1 |
| paper@512 | 76 | 45.2 | 22.8 | 92 | 3,764 | 63.7k | 1,520 | 256.6 |
| paper@512 + ledger | 88 | 52.4 | 21.8 | 80 | 4,167 | 67.3k | 2,630 | 334.9 |

McNemar (test): 4096 vs 512 +54/−12 p=1.7e-7; 512 vs none +21/−49 p=0.001; 4096 vs none +35/−21
p=0.081; ledger vs 512 +33/−21 p=0.134; ledger vs none +26/−42 p=0.068; 4096 vs ledger +48/−18 p=0.0003.

## 5. Code changes to ACON (all in `acon-main/`)

- `src/productive_agents/llm.py`: `OllamaModel` (OpenAI-compatible client); per-request log when
  `ACON_REQ_LOG` is set (prompt/completion tokens, seconds; uses `time.clock_gettime(CLOCK_REALTIME)`
  because AppWorld freezes `time.time()` to 2023-05-18).
- `src/productive_agents/agents/utils.py`: model-name routing to Ollama.
- `src/productive_agents/ctxopt/state_ledger.py` (new): deterministic ledger — variables (AST),
  credentials (JWT/email regex), APIs called with literal args, nonexistent APIs, recent errors.
- `src/productive_agents/ctxopt/history_optimizer.py`: `state_ledger: true` config switch appends
  the ledger to the summary and logs it in `prompt_args["state_ledger"]`. Off by default.
- Configs `experiments/appworld/configs/context_opt/ollama_qwen2.5_7b_c16k_*.yaml`; runner
  `run_arms_c16k.sh <split> tag=cfg|NONE ...` (env `TASKS_FILE`, `MEMCAP_KB`); task subsets in
  `data/datasets/{dev_half_a,dev_half_b,dev_uni_m1,dev_uni_m2,tn_part}.txt` on the machines.
- Our code matches Microsoft upstream except the above and the agent prompt (diffed 2026-10-31).

## 6. Lab machines and operations

| | M1 | M2 |
|---|---|---|
| SSH | `cse-sdpl@100.79.174.22` | `iiitdmk@100.105.182.53` |
| Password (lab, given by user) | `1234` | `1234` |
| GPU | RTX 2000 Ada 16 GB | RTX 2000 Ada 16 GB |
| Code | `~/acon-main` (venv `.venv`, Python 3.12.3) | copy of M1; `/home/cse-sdpl` is a symlink to `/home/iiitdmk` so hard-coded paths work |
| Ollama | 0.20.0 system service | official 0.20.0 (user installed); old community snap disabled (`sudo snap enable ollama` reverses) |
| Desktop | X11, `DISPLAY=:1`, `XAUTHORITY=/run/user/1000/gdm/Xauthority` | Wayland, `WAYLAND_DISPLAY=wayland-0` |

- Access is over Tailscale. If both machines time out, first check the Mac:
  `tailscale status --json` → `Self.Online`. Do not run `tailscale down/up` from the CLI (it logged
  the Mac out on 2026-10-31); ask the user to toggle it from the menu bar.
- Suspend is masked on both (`systemctl is-enabled suspend.target` → masked). Machines have been
  powered off by other people twice; runs skip finished tasks, so just relaunch.
- Keep-alive while running: `gnome-session-inhibit` + full-screen `notice_c16k.sh` (env PROGRESS,
  TOTAL, PATTERN, DURATION) + `~/Desktop/DO_NOT_SHUTDOWN_experiment_running.txt`. On M2 also set
  `idle-delay 0` and `lock-enabled false` during runs; **restore to 900 / true afterwards**.
- Per-task process with `ulimit -v` memory cap; some tasks (`df61dc5_1`, `57c3486_3`) exceed 20 GB,
  so use `MEMCAP_KB=41943040` (40 GB).
- M2 GPU driver fix (2026-10-31): installed `linux-modules-nvidia-580-7.0.0-34-generic`; also
  completed the 7.0.0-38 module so reboots keep working.

### Pitfalls that already cost time
- **zsh `$H:a`**: in this shell `"$H:acon-main"` expands `:a` as a modifier and rsync writes to a
  local folder. Always write `"${H}:acon-main/..."`.
- **pgrep self-match**: a command line containing the pattern matches itself. Send scripts over
  stdin (`ssh host 'bash -s' <<'EOF'`) or use `[r]egex` brackets.
- **Message logs double-count**: `llm_history.json` repeats the preserved last turn around each
  compression; count steps from `env_history.json`.
- **Ollama silently truncates** at `num_ctx`; check `prompt_tokens` in the request log.
- Running two jobs on one GPU makes runs diverge and slows both. One queue per machine.
- Safety checks block `rm -rf` inside remote heredoc scripts and running downloaded install
  scripts as root; ask the user instead.
