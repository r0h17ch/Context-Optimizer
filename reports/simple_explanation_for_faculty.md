# My final-year project in simple words

## The problem

AI assistants can now do multi-step jobs on their own, for example "like all the songs on my
Spotify playlist". To do this, the assistant keeps a running record of everything it has done
so far. On long jobs this record becomes very long, which makes the assistant slow and
expensive, and can also confuse it.

A common fix is to **shrink the record** from time to time: another AI writes a short summary of
what happened, and the assistant continues from that summary.

## The base paper

My base paper is **ACON** (ICML 2026, a top machine-learning conference). It gives the summarizer
a carefully tuned set of instructions (a "guideline") for writing these summaries. The paper
reports that this keeps the assistant just as accurate while using fewer words.

## What I did

1. I ran ACON's own code on free, open-source AI models on two college lab computers, instead of
   the paid models used in the paper. In total I ran about 2,000 tasks on a standard benchmark
   (AppWorld, simulated apps such as Spotify and Venmo).
2. I first checked that I could reproduce the paper's result at its own settings.
3. I then tested a setting that shrinks the record more often, which saves even more words.
4. I studied the logs to find out why the results changed.
5. I tried a fix and tested everything on 168 tasks that I never looked at while developing,
   so the final numbers are fair.

## What I found

| Setting (168 unseen tasks) | Tasks solved |
|---|---|
| No shrinking | 61.9% |
| ACON, at the paper's setting | **70.2%** |
| ACON, shrinking more often | **45.2%** |
| ACON, shrinking more often, with my fix | 52.4% |

- **ACON works at the paper's own setting.** It solved more tasks than no shrinking and used about
  30% fewer words.
- **ACON breaks when it shrinks the record more often.** Success drops from 70.2% to 45.2%. This
  drop is far too large to be chance (statistical test, p < 0.0000002).
- **I found the reason.** ACON's tuned guideline tells the summarizer to delete error messages.
  So the summaries usually leave out what went wrong (only 32% of the time they mention a
  failure, against 87% for a simple, untuned guideline). When the record is shrunk often, the
  assistant keeps forgetting which attempts already failed. It repeats the same mistakes, such
  as calling app functions that do not exist, and runs out of steps.
- **My first fix helped only a little.** I added a short, automatic list of key facts and recent
  errors to every summary. On the tasks I used while developing it, it fully solved the problem.
  On the unseen tasks it improved success from 45.2% to 52.4%, which is not enough to be sure it
  is a real improvement.
- I also checked one more claim of the paper (that shrinking two kinds of information together
  hurts badly). That claim did not hold with open-source models.

## What it means

The project shows a clear, measured weakness of the base paper: its tuned instructions are
fragile, and I can explain exactly why. This is the main contribution so far.

## Next step

Test the explanation directly: give the summarizer one extra instruction to keep error
messages, and check whether that alone brings the success rate back up. If it does, this simple
change becomes my improvement to ACON.

## What would help

- Access to a better GPU, to repeat the key tests and to try a larger AI model.
- Your advice on whether to aim for a workshop paper first.
