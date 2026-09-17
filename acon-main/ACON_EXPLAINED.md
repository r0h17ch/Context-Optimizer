# ACON, Explained — and How SAC Could Fit

Learning notes for the FYP. Plain English, written from reading the actual code in this
repo rather than from the paper. Findings from our runs are logged at the bottom and
updated as results come in.

---

## Part 1 — The problem ACON solves

An LLM agent doing a multi-step task builds up a conversation:

```
system prompt
user:      "Here are the rules. Your task is: follow all classical artists with 22+ followers"
assistant: print(apis.api_docs.show_app_descriptions())
user:      [a 3000-token JSON blob listing every app]
assistant: print(apis.spotify.show_artists())
user:      [another 4000-token JSON blob]
assistant: ...
```

Every step appends to this. By step 20 the prompt is enormous. Three things go wrong:

1. **Cost** — you re-send the whole history on every single call.
2. **The context window fills up** and the model starts losing the early instructions.
3. **Signal gets buried** — the one useful line is drowned in thousands of tokens of JSON.

**ACON's idea:** put a second, smaller LLM in the loop whose only job is to *rewrite the
context into something shorter that keeps the important parts*. That second model is the
**compressor**. The model doing the actual task is the **agent**.

In our setup:
- agent = `qwen2.5:14b`
- compressor = `qwen2.5:7b`

ACON compresses in two different places, and they are independent:

| Optimizer | What it squashes | When it runs |
|---|---|---|
| **History optimizer** | The whole conversation so far | *Before* asking the agent for its next action |
| **Observation optimizer** | One tool output that just came back | *After* the environment returns a result |

You can enable either, both (`unified`), or neither (the baseline).

---

## Part 2 — The main loop

This is `UnifiedAgent.run()` in
[src/productive_agents/agents/unified_agent.py](src/productive_agents/agents/unified_agent.py#L302).
Stripped to the essentials:

```python
while not done and n_iter < max_iter:
    user_prompt = self.build_prompt(env)            # 1. build the next user message
    self.memory_manager.add_user_prompt(user_prompt)

    if self.memory_manager.do_history_optimization:  # 2. HISTORY COMPRESSION
        self.memory_manager.optimize_history(task=task_desc, opt_args={})

    prompt = self.memory_manager.get_conversation_history(exclude_system=True)
    llm_output = self.forward(prompt)                # 3. agent LLM generates code
    action = llm_output.action

    self.memory_manager.add_assistant_response(raw_response)

    obs, reward, done, info = env.step(action)       # 4. run the code in AppWorld

    if self.memory_manager.do_observation_optimization:   # 5. OBSERVATION COMPRESSION
        refined_obs = self.memory_manager.optimize_observation(
            task=task_desc, observation=obs, opt_args={})
        env.observation = refined_obs                # the SHORT version is what gets stored
```

Read it as a cycle: **build prompt → (compress history) → agent thinks → execute code →
(compress result) → repeat.**

The key detail in step 5: the optimizer overwrites `env.observation`. The next loop
iteration builds its prompt from that, so the agent **never sees the raw output** — only
the compressed version. Compression is destructive and one-way. If the compressor throws
away a detail the agent needed three steps later, it is gone for good. That is the central
risk of the whole approach, and it is why the prompts push so hard on "keep anything that
looks like an ID or a result."

Both compression calls are wrapped in `try/except` that only logs a warning. **If the
compressor fails, the run silently continues uncompressed.** Worth knowing — a broken
compressor does not crash the run, it just quietly does nothing. (This is exactly how our
first runs looked "fine" while doing no compression at all.)

---

## Part 3 — File map

### The parts you will actually touch

| File | What it does |
|---|---|
| [experiments/appworld/run_all.py](experiments/appworld/run_all.py) | Entry point. Parses args, loads YAML configs, loops over tasks, writes `experiment_summary.json`. |
| [experiments/appworld/run.py](experiments/appworld/run.py) | Runs **one** task: builds env + agent, calls `agent.run()`, saves results. |
| [src/productive_agents/agents/memory.py](src/productive_agents/agents/memory.py) | **The heart of ACON.** Owns the conversation and decides when/how to compress. |
| [src/productive_agents/ctxopt/history_optimizer.py](src/productive_agents/ctxopt/history_optimizer.py) | Summarizes the conversation. |
| [src/productive_agents/ctxopt/obs_optimizer.py](src/productive_agents/ctxopt/obs_optimizer.py) | Summarizes a single tool output. |
| [src/productive_agents/ctxopt/base.py](src/productive_agents/ctxopt/base.py) | Shared helpers for both: token counting, Jinja templates, output parsing. |
| [src/productive_agents/llm.py](src/productive_agents/llm.py) | Talks to models. `OllamaModel` is the class we use. |
| `experiments/appworld/configs/context_opt/*.yaml` | Which compressor, which mode, what thresholds. |
| `experiments/appworld/prompts/context_opt/*.jinja` | The actual instructions given to the compressor. |

### Supporting cast

| File | What it does |
|---|---|
| [src/productive_agents/agents/unified_agent.py](src/productive_agents/agents/unified_agent.py) | Generic agent loop shared across benchmarks. |
| [src/productive_agents/agents/appworld/agent.py](src/productive_agents/agents/appworld/agent.py) | AppWorld specifics — the big few-shot prompt, how to extract code from a reply. |
| [src/productive_agents/agents/appworld/config.py](src/productive_agents/agents/appworld/config.py) | Config dataclass, including `co_config`. |
| [src/productive_agents/env/appworld/env.py](src/productive_agents/env/appworld/env.py) | Wraps AppWorld: `step(code)` → `(obs, reward, done, info)`. |
| [src/productive_agents/agents/base.py](src/productive_agents/agents/base.py) | Abstract base classes. |

Everything under `env/officebench/` and `agents/officebench/` is a **different benchmark**
we are not using. Ignore it.

---

## Part 4 — How history compression actually works

In `MemoryManager.optimize_history()`
([memory.py](src/productive_agents/agents/memory.py#L355)). The conversation is stored as
`llm_history`, a **list of sessions**, where each session is a list of messages.

**Step 1 — protect the recent turns.** `preserve_last_k_turns` (we use 1) turns are pulled
out and exempted from compression. The agent always sees its most recent action and result
verbatim, because that is what it needs to decide the next move.

**Step 2 — check if it is even worth it.**

```python
n_tokens = self.count_tokens(history_text)
return n_tokens > self.history_summarization_threshold
```

Everything *except* the preserved turns and the first user message gets counted. If it is
under the threshold, the function returns and nothing happens.

> **This is the single most important knob, and it is where we lost our first two runs.**
> The shipped default is 4096. On short AppWorld tasks the history text is 57-700 tokens,
> so the check never passed and the compressor never ran once. See Finding 2.

**Step 3 — summarize.** The history text goes into
[prompt_history_v2.jinja](experiments/appworld/prompts/context_opt/prompt_history_v2.jinja),
which asks for a structured summary under `### REASONING` and `### COMPLETED` headings. The
compressor LLM answers, and `parse_output()` keeps everything after `# History Summary`.

**Step 4 — rebuild the conversation.** This is the clever bit:

```python
self.start_new_session()
self.add_system_prompt(self.system_prompt, new_session=True)
self.add_user_prompt(user_prompt, new_session=True)   # task + <HISTORY_SUMMARY>...</HISTORY_SUMMARY>
for msg in preserved_turns:
    ...  # the last k turns, verbatim
```

A **brand new session** is started containing only:

```
system prompt
user:  [original task] + <HISTORY_SUMMARY> ...the summary... </HISTORY_SUMMARY>
       [the last k turns, untouched]
```

`get_conversation_history()` returns only the *current* session, so the next agent call
sees this short version. The old session still sits in `llm_history` for logging, but is
never sent to the model again.

**`history_summary_rule`** controls what the summary is attached to:
- `reset` (ours) — rebuild from `first_user_prompt`; each new summary **replaces** the old.
- `accumulate` — summaries pile up on top of each other, so context creeps back up.

The code comments flag `accumulate` as a known wart. `reset` is the sane default.

---

## Part 5 — How observation compression works

Simpler. In `ObservationOptimizer.process()`
([obs_optimizer.py](src/productive_agents/ctxopt/obs_optimizer.py#L80)):

1. `check_summarization_needed(observation)` — is this single output longer than
   `obs_summarization_threshold`? (Default 1024; we lowered it to 256.)
2. If yes, render [prompt_user.jinja](experiments/appworld/prompts/context_opt/prompt_user.jinja)
   with the task, the observation, and the conversation so far — **the task is included on
   purpose**, so the compressor knows what to keep.
3. Ask the compressor, keep everything after `# Refined Observation`.
4. Return it; the caller assigns it to `env.observation`.

History vs observation, side by side:

| | History optimizer | Observation optimizer |
|---|---|---|
| Input | Whole conversation | One tool output |
| Frequency | Only past threshold | Every oversized output |
| Effect | Rebuilds the session | Replaces one message |
| Fails when | Drops a fact needed later | Mangles a single result |

---

## Part 6 — Config and wiring

A run is assembled from a YAML file passed as `--co_config_path`:

```yaml
type: "history"                        # history | obs | unified   <-- REQUIRED
model: "ollama/qwen2.5:7b"             # the COMPRESSOR (agent comes from --model_name)
compressor_type: "full"
prompts:
  prompt_system: "system_prompt"
  prompt_history_user: "prompt_history_v2"
history_summarization_threshold: 512
preserve_last_k_turns: 1
history_summary_rule: "reset"
temperature: 0.0
```

`run_all.py` loads this into `co_config`, hands it to the agent config, and
`MemoryManager.__init__` dispatches on `type`:

```python
co_type = co_config.get("type", None)
if   co_type == "obs":     self.obs_optimizer = obs_cls(...);  self.do_observation_optimization = True
elif co_type == "history": self.history_optimizer = hist_cls(...); self.do_history_optimization = True
elif co_type == "unified": # both
else: raise ValueError(f"Unknown context optimization type: {co_type}")
```

That last line is where the shipped qwen config died. See Finding 1.

Two separate model settings that are easy to confuse:

- `--model_name qwen2.5:14b` → the **agent**
- `model:` in the YAML → the **compressor**

They are independent. A big agent with a small cheap compressor is the whole point.

---

## Part 7 — Findings log

Updated as runs complete. These are our own results, not the paper's.

### Finding 1 — The shipped qwen config was broken

`configs/context_opt/ollama_qwen2.5_14b_history.yaml` looked like this:

```yaml
model: "ollama/qwen2.5:14b"
baseline_strategy: "none"
optimizer_type: "history"     # <-- wrong key name
threshold: 6000               # <-- wrong key name
```

`MemoryManager` reads `type`, not `optimizer_type`. With `type` missing it raises
`ValueError: Unknown context optimization type: None` on the first task, instantly. That is
why the earlier `paper_replication_run` left an empty task folder and no summary — the run
never started. Fixed; original kept as `.bak`.

**Lesson:** the config keys are not validated. A typo'd key is silently ignored and you get
either a crash or, worse, a run with compression quietly disabled.

### Finding 2 — Default thresholds never fire on this benchmark

With the stock threshold of 4096, the compressor ran **zero** times across a whole run.
Measured history lengths on AppWorld tasks:

| Task length | History text (tokens) | Fires at 4096? | Fires at 512? |
|---|---|---|---|
| 4 iterations | 57 - 221 | no | no |
| 12 iterations | 627 - 681 | no | **yes** |

The paper's thresholds are tuned for much longer-horizon tasks. On short AppWorld tasks
they mean ACON is switched off while *appearing* to be on — and because compression
failures are caught and logged as warnings, nothing tells you.

We lowered history to **512** and observation to **256**. Compression then fired 31 times
in one 8-task arm.

This is a genuine contribution for the report: *applying ACON to short-horizon tasks
requires retuning the thresholds, or the method is a no-op.* We are running an
`acon_hist_q7b_t4096` arm at the paper default specifically to quantify this.

### Finding 3 — Ollama's context window silently truncates

`ollama ps` shows both models served with `CONTEXT 4096`. In the baseline, cumulative agent
input reaches ~100k tokens on long tasks, and individual requests exceed 4096. Ollama
**truncates silently** — no error, no warning.

Consequence: our baseline is handicapped in a way that is not about context length per se.
Any ACON-vs-baseline gap partly measures "ACON kept the prompt under the window", not
"compression helps". We cannot raise the window because the 14b agent (9.7 GB) and 7b
compressor (4.9 GB) already use 14.6 GB of the 16 GB card.

**This is the biggest limitation on our numbers and must be stated in the report.**

### Finding 4 — Agent-generated code can OOM the whole machine

A 38-task run grew to **59 GB RSS** and was OOM-killed by the kernel:

```
Out of memory: Killed process 138586 (python)
total-vm:69715844kB, anon-rss:59016600kB
```

**First diagnosis was wrong.** I initially assumed `run_all.py` leaked across tasks. After
isolating tasks into separate processes, the truth is narrower and more interesting: a
*single* task, `771d8fc_1`, does it on its own, reproducibly, on iteration 7:

```
ERROR:productive_agents.env.appworld.env:Error executing action:
ERROR:productive_agents.agents.unified_agent:Error during execution:
...
MemoryError
```

The empty error message is the giveaway — `MemoryError` stringifies to `""`. The agent wrote
Python that allocates unbounded memory (an enormous list or range), and **AppWorld executes
agent-generated code in the harness's own process**, so there is nothing between a bad
`action` and the host's RAM.

Two things follow:

1. **Security/robustness note for the report.** The sandbox blocks OS-level modules, but not
   resource exhaustion. Any agent that emits `[0] * 10**10` takes down the machine. On a
   shared lab box that affects other users.
2. **It is a property of the task, not of ACON.** It shows up in the baseline, so it is not
   caused by compression. But it will recur in every arm that reaches this task.

Fix: one process per task with a 20 GB `ulimit -v` cap. The bad task now dies alone with
`rc=1` after ~2 minutes and the arm continues; per-task RSS is 0.8 GB instead of 59 GB.
`771d8fc_1` will show up as a missing result in every arm — that is expected and should be
reported as such, not quietly hidden.

### Finding 5 — `max_iter` dominates the success rate

| `max_iter` | Baseline success |
|---|---|
| 12 | 12.5% (1/8) |
| 30 | 36.8% (7/19) |

Several tasks need 14-21 iterations. Cap below that and you manufacture failures that have
nothing to do with context optimization. **Do not compare numbers across different
`max_iter` settings.**

### Finding 6 — Pilot run (8 tasks, `max_iter` 12)

| Arm | Solved | Agent input tokens | Compression calls |
|---|---|---|---|
| baseline | 1/8 (12.5%) | 269,935 | 0 |
| ACON history, 7b | 2/8 (25.0%) | 232,749 | 31 |
| ACON observation, 7b | 1/8 (12.5%) | 261,422 | 6 |

Compression ratio: **0.149** for history (34,320 → 5,117 tokens over 31 calls), 0.125 for
observation.

**How to read this honestly:** the token reduction (−13.8%) and the compression ratio are
real, aggregated over many calls. The success-rate difference is **one task out of eight** —
that is noise, not evidence. Do not write "ACON doubled success rate". The observation arm
only fired 6 times, so it is closer to untested than to a negative result.

### Finding 7 — Context truncation causes degenerate repetition

Caught live while watching the baseline arm. On task `60d0b5b_1` the logs showed:

```
Token usage - Input: 4020, Output: 868, Total: 100359
Token usage - Input: 4020, Output: 868, Total: 105247
```

Identical input and output sizes, repeating, at ~48 seconds per request (normal requests
take 7-17s). Input is pinned at ~4020 tokens, right against the 4096 ceiling from
Finding 3.

What is happening: once the prompt hits the window limit, Ollama truncates the front of it.
The agent loses the early instructions, produces a long confused reply, that reply makes
the next prompt hit the ceiling again, and it locks into a loop — burning all 30 iterations
at ~48s each, roughly 24 minutes on a single task it can never finish.

This connects Findings 3 and 5: the reason some tasks consume every iteration is not that
they are intrinsically hard, but that truncation has destroyed the instructions. **It also
predicts where ACON should help most** — compression keeps the prompt under the ceiling, so
truncation never triggers and the loop never starts. If the ACON arms show fewer
30-iteration tasks than the baseline, this is the mechanism.

Worth checking in the final data: count tasks hitting `max_iterations_reached` per arm.

### Finding 8 — The split has 39 tasks, not 38

Small but it would have made our denominators wrong. `wc -l` on
`data/datasets/train_history_tiny.txt` reports 38 because the file has **no trailing
newline**; `load_task_ids("train_history_tiny")` returns 39. Always trust the loader.

### Finding 9 — Baseline, full split, `max_iter` 30

| Metric | Value |
|---|---|
| Tasks completed | 38 / 39 (`771d8fc_1` died on the Finding 4 MemoryError) |
| Solved | **12 — 31.6%** |
| Agent input tokens | 2,972,376 |
| Agent output tokens | 190,214 |
| **Tasks hitting `max_iter` (30)** | **26 of 38 — 68%** |
| Wall clock | 2h 02m for the final 19 tasks |

The headline number for the report is that **68% of tasks burn every available iteration
without finishing.** Combined with Finding 7 (input pinned at ~4020 tokens against the 4096
ceiling, identical responses repeating at 48s each), the picture is that most baseline
failures are not the model being incapable — they are the model being lobotomised by
context truncation partway through and then looping.

**This is the number to watch in the ACON arms.** If compression works as advertised, it
keeps the prompt under the ceiling, truncation never fires, and the `hit_max_iter` count
should drop. That comparison is more diagnostic than success rate, because it isolates the
mechanism instead of the outcome, and it is far less sensitive to one or two lucky tasks.

### Finding 10 — The lab machine auto-suspends mid-run

The workstation went dark at 18:33 and stayed unreachable overnight. The journal was
unambiguous:

```
systemd-logind: The system will suspend now!
NetworkManager: manager: NetworkManager state is now ASLEEP
```

Ubuntu's GNOME default is `sleep-inactive-ac-type = suspend` after **7200 s** idle. A
benchmark driven over SSH does not count as user activity, so after two hours of nobody
touching the desktop the machine suspended itself and took Wi-Fi and Tailscale down with it.

The run itself survived: `nohup`-ed processes are frozen on suspend and resume on wake, so
the driver picked up exactly where it stopped. What was lost was ~15 hours of wall-clock.

Fixed on 2026-09-17:

```bash
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type nothing
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
sudo systemctl enable tailscaled
```

To undo later (it is a shared lab machine):
`sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target` and
`gsettings reset org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type`.

**Lesson for any long run on a desktop machine:** check power settings before launching, and
sync results off the box after every arm, not at the end.

### Finding 11 — ACON arms, full split

_In progress. `acon_hist_q7b` started 17:38. Then `acon_hist_q7b_t4096` (paper default
threshold) and `acon_obs_q7b`. Scope was cut from six arms to four to fit the deadline;
the unified and `llama3.1:8b` compressor arms were dropped._

---

## Part 8 — Where SAC could fit

### What SAC actually is

SAC (*Autoencoding-Free Context Compression via Contextual Semantic Anchors*, ICLR 2026)
compresses at a completely different level from ACON.

From [SAC/model/modeling.py](../SAC/model/modeling.py), `CompressLLM.compress()` takes the
input embeddings and produces `compress_token` — a tensor of **hidden states**, roughly
`seq_len / compress_ratio` of them. Those vectors are fed into the decoder directly as
embeddings / `past_key_values`.

The difference in one line:

| | ACON | SAC |
|---|---|---|
| Compresses into | **Text** (a written summary) | **Vectors** (soft tokens) |
| Mechanism | Prompt a second LLM | LoRA-adapted encoder pass |
| Output readable? | Yes, you can read the summary | No, they are just numbers |
| Ratio | Emergent (~6.7x for us) | Set explicitly (5x, 15x) |
| Needs training? | No, prompting only | **Yes** — LoRA adapter |
| Interface | HTTP, any text API | In-process, needs embedding access |

### Answering the question: would SAC be a third model alongside the other two?

**Not as a drop-in, no.** This is the part that matters, so it is worth being precise.

ACON's compressor is called through
[`llm.py`](src/productive_agents/llm.py)'s `OllamaModel.generate(prompt) -> str`. Text in,
text out, over HTTP. Everything downstream assumes a string: `parse_output()` looks for a
`# History Summary` marker, the result is spliced into a `<HISTORY_SUMMARY>` tag inside a
user message.

**SAC does not return text.** It returns hidden-state tensors that only mean anything to
the specific decoder they were produced for. You cannot send a tensor through Ollama's HTTP
API, and you cannot paste one into a prompt string. So "just run SAC as a third model next
to qwen 14b and 7b" is not possible without changing the interface.

There are two further mismatches:

1. **Tokenizer/model coupling.** SAC's compressed vectors are tied to its base model
   (Llama-3.2-1B in our reproduction). Feeding them to `qwen2.5:14b` is meaningless — the
   embedding spaces are unrelated.
2. **Domain.** SAC is trained on MRQA (single-passage QA). Agent trajectories — code,
   JSON dumps, tool errors — are a different distribution entirely.

### Three ways to actually do it

**Option A — SAC-style compression, ACON-style interface (easiest, dishonest if mislabelled)**

Keep the text interface; just change the prompt so the compressor emits something very
terse and anchor-like. Fits the existing code with no changes.

*This is not SAC.* It borrows the intuition (anchor on the semantically load-bearing bits)
but none of the mechanism. Only acceptable if you describe it as "SAC-inspired prompting",
never as "we integrated SAC".

**Option B — Replace the agent's serving stack (the real integration)**

Drop Ollama for the agent; load the model in-process with HF `transformers` so you can pass
`inputs_embeds`. Then:

- run SAC's encoder over the history to get `compress_token`
- prepend those vectors to the agent's input embeddings
- add a `type: "sac"` branch in `MemoryManager.__init__`
- add a `SACOptimizer` in `ctxopt/` that returns tensors rather than `str`
- teach the memory manager to carry a tensor slot instead of splicing a string

Blockers to be honest about: SAC must be retrained (or at least LoRA-adapted) for the agent
model, VRAM is already at 14.6/16 GB, and `MemoryManager` assumes strings throughout. This
is a multi-week change, not a weekend one.

**Option C — Compare them instead of combining them (recommended for the report)**

Do not integrate. **Benchmark them as two answers to the same question.** You already have
both halves working:

- SAC reproduced on MRQA within 0.03-0.15 F1 of the paper
- ACON running end-to-end on AppWorld with a measured 0.149 compression ratio

The report writes itself as: *two context-compression paradigms — textual summarization vs.
latent anchors — evaluated on their own benchmarks, with an analysis of why they are not
interchangeable and what an integration would require.* That analysis **is** the
contribution, and Option B becomes your future-work section with a credible plan.

### Suggested future-work framing

1. **Unify the benchmark.** Neither method has been tested on the other's task. Running SAC
   on agent trajectories, or ACON on MRQA, would say something new.
2. **A common compression-ratio axis.** SAC sets ratio explicitly (5x/15x); ACON's emerges
   (~6.7x). Plotting task success against ratio on one chart is the natural comparison.
3. **Hybrid.** ACON for history (needs to stay human-readable and survive session rebuilds),
   SAC for observations (single blobs, no cross-step reasoning needed). Architecturally the
   most plausible combination, since observations are consumed once.
4. **Fix the interface.** A compressor API returning *either* text or embeddings would make
   the two swappable. That refactor is a contribution in itself.

---

## Part 9 — Running things

```bash
cd /home/cse-sdpl/acon-main/experiments/appworld
export PYTHONPATH=/home/cse-sdpl/acon-main/src

# one arm
/home/cse-sdpl/acon-main/.venv/bin/python run_all.py \
    --split train_history_tiny --model_name qwen2.5:14b --tag my_run \
    --co_config_path configs/context_opt/ollama_qwen2.5_7b_history.yaml --max_iter 30

# whole matrix (memory-safe, process per task)
./run_acon_full_test.sh

# regenerate RESULTS.md from whatever has finished
/home/cse-sdpl/acon-main/.venv/bin/python aggregate_results.py
```

**Sanity checks before trusting any run:**

1. `grep -c "criterion check" run_logs/<tag>.log` — is the compressor being consulted?
2. `find outputs/<...> -name history_optimizer_history.json -size +3c` — did it produce
   anything? An empty `[]` means the threshold was never crossed.
3. `ollama ps` — are both models actually on the GPU, and what is `CONTEXT`?
4. `free -g` — memory climbing toward the ceiling means the leak is back.

A run that "completes successfully" with zero compression calls is the failure mode to
watch for. It looks identical to a successful run in every summary metric.
