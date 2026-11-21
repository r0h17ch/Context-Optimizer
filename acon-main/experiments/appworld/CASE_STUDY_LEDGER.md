# Why ACON's optimized guideline collapses at threshold 512, and why the ledger fixes it

AppWorld **dev** (57 tasks), agent qwen2.5:14b, compressor qwen2.5:7b, num_ctx 16k, max 30 steps,
temperature 0. Regenerate the numbers with `analyze_dev.py` and `analyze_cfr.py`; the
per-step counts below come from `env_history.json` (executed steps only; the message logs
repeat the preserved last turn around every compression and over-count).

## 1. The effect

| History compression | Success | Steps | Hit 30-step cap | Compressions |
|---|---|---|---|---|
| none | 69.6% | 18.0 | 17 | 0 |
| naive guideline (D.5) @512 | 69.6% | 18.4 | 17 | 397 |
| paper guideline (D.7) @4096 | 73.2% | 17.1 | 15 | 39 |
| paper guideline (D.7) @512 | **42.9%** | 22.2 | **32** | 510 |
| paper guideline (D.7) @512, repeat run | 44.6% | 21.7 | 31 | 465 |
| paper guideline (D.7) @512 **+ state ledger** | **73.2%** | 17.7 | 15 | 685 |

Paper@512 vs paper@4096: +20 / −3, McNemar p = 0.0005. Ledger vs paper@512: +22 / −5,
p = 0.0015 (vs the repeat run: p = 0.0025). The two paper@512 runs agree on 55 of 56 tasks.

21 tasks fail in **both** paper@512 runs and are solved with the ledger; they are the cases below.

## 2. What it is not

| Hypothesis | Test | Result |
|---|---|---|
| Lost exact values (tokens, ids, emails) | CFR over facts the uncompressed run later used | paper@512 keeps *more* (34%) than naive@512 (28%), which still succeeds; per-task CFR vs success r = −0.08 |
| Lost access token | token present in compressed history kept verbatim | naive@512 keeps 0% and succeeds; the agent reuses the `access_token` *variable* in the REPL |
| Lost API list from `api_docs` | API names from `show_api_descriptions` kept in summary | every arm keeps 1–3% |
| Agent stops writing code | steps that are syntax errors from prose | 2–5% in every arm |
| Exact action loops | identical consecutive actions | 8–15% in every arm |

## 3. What it is: the guideline deletes failures

When the history being compressed contains failed executions, does the LLM-written summary say so?

| | naive (D.5) | paper (D.7) |
|---|---|---|
| @512 | **87%** | **32%** (repeat: 30%) |
| @4096 | 51% | 35% |
| @512 with ledger (LLM part only) | – | 38% |

The two prompts differ exactly here. D.5's priorities include *"Include key info from errors or
failed attempts to prevent repeated mistakes."* D.7 says *"Delete tool/API log output"*,
*"Replace repetitive step logs with one summarizing phrase"* and keeps a ≤5-bullet COMPLETED list.

Consequence, measured on executed steps: calls to APIs that do not exist are **46% / 45%** of
steps in the two paper@512 runs against **36–37%** in every other arm, so the agent keeps
retrying approaches that already failed and runs into the step cap.

Why only at 512: there the history is compressed ~9 times per task (510 compressions / 57),
so the record of what failed is erased again and again; at 4096 it happens less than once per
task (39 / 57), so the same flaw rarely bites.

Why the ledger fixes it: its `RECENT ERRORS` and `APIS THAT DO NOT EXIST` sections restore the
failure record deterministically, even though its LLM-written part is just as forgetful (38%).
With it, nonexistent-API calls drop to **32%** of steps, below the uncompressed run.

## 4. Worked example: `d4e9306_2`

*"Follow artists of all the songs and albums I have ever liked on Spotify."* (solved without
compression; failed in both paper@512 runs; solved with the ledger in 8 steps.)

By the third compression **every** step so far had failed. The paper-guideline summary:

```
2. VARS
access_token=<token>, artist_ids=set()
4. COMPLETED
- Retrieved access token.
- Fetched user playlists.
- Extracted liked songs and albums.
- Collected unique artist IDs.
5. GUARDRAISLS
Ensure correct API usage and permissions for following artists.
```

It reports progress that never happened (`artist_ids` is empty) and no error. The agent then
logs in again and restarts the same approach on steps 10–16 and never finishes.

The ledger appended at the same point:

```
APIS ALREADY CALLED (count, last literal arguments):
- apis.spotify.search_songs x1 (query='is:liked')
- apis.spotify.search_albums x1 (query='is:liked')
APIS THAT DO NOT EXIST (do not call again; check api_docs instead):
- apis.spotify.get_liked_tracks_and_albums
- apis.spotify.get_liked_tracks
RECENT ERRORS:
- Exception: No API named 'get_liked_tracks_and_albums' found in the spotify app.
```

With the failures visible, the agent switches approach and completes the task.

## 5. Caveats

- One benchmark, one agent/compressor pair (qwen2.5 14B / 7B); the paper's guidelines were
  optimized for gpt-4.1, so part of the failure may be a 7B compressor following D.7 literally.
- "Admits failure" is a keyword test on the LLM-written summary (fail, error, exception,
  not found, unauthorized, 401/422, invalid, unable, could not).
- The mechanism is supported by correlation across arms, not yet by intervention. The
  decisive test is the ledger ablation: **errors-only ledger** (RECENT ERRORS + nonexistent
  APIs) vs **values-only ledger** (VARIABLES + CREDENTIALS). If errors-only recovers the
  accuracy, the mechanism is established.
- Related: *Toward Reliable Context Compression for Long-Horizon Agents* (arXiv 2608.06503)
  reports blocked actions and repeated exploration after compression on AppWorld.
