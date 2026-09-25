# Voice Phase 1: results

DRAFT: numbers are filled in from `results/voice/` as the runs finish.

## What was built

### `tez.stream`: a streaming decision session in the runtime

```python
from tez import Tez
from tez.stream import Policy

tez = Tez(backend="inproc:gemma-4-12b-it-Q4_K_M.gguf", template="gemma4", schemas="data/voice/theioma/schema.yaml",
          inproc={"lib": "C:/temp/llamacpp", "n_ctx": 2048, "n_seq_max": 8, "n_outputs": 64})
session = tez.stream(schema="theioma-voice", on_action=execute, on_undo=retract).load()   # prefix evaluated once

u = session.feed("open")                 # a complete word
u = session.update("open no")            # or a recogniser's text as it grows: the last word may still change
u.decision, u.confidence, u.committed, u.action, u.stable, u.timings["decide_ms"]
u = session.end()                        # deferred (R2) actions act, R3 becomes an approval request, none hands off
session.undo()                           # retract the last action
session.reset()                          # next utterance
```

- One session answers one choice question of a schema, question first (instructions and options, then the transcript
  as the state), after every word or recogniser step.
- In process (`InprocBackend`), `InprocStreamReader` keeps per prompt three sequences: the constant prefix (evaluated
  once, never touched again), the committed transcript, and a scratch copy that receives the new words plus the answer
  tail for each decision. The word is committed after the decision and the dispatch, in the gap before the next word
  (`commit="gap"`), or folded into the next decision's decode (`commit="fold"`: one forward pass per step). Nothing is
  ever rolled back, so hybrid Qwen3.5 models work the same way. A revised transcript rebuilds from the prefix.
- Answers are read by `Tez.decide_question` itself from the stream's reads, so letters, probes, blends, `__none__`,
  fitted temperatures and `Tez.unfitted_temperature` apply exactly as in a `/v1/systemone` request (tested:
  probabilities equal to a whole-prompt `decide_question` to 1e-6 on the stub, per word, for letters, probes, an
  abstaining question, temperature auto/off/number, hybrid models and partial words).
- Other backends (llama-server over HTTP, `FakeBackend`) work too: the whole prompt per word, llama-server's prompt
  cache reusing the prefix.
- The early-commit policy of `experiments/stream_policy.py` is data: the schema's `actions:` block gives each option a
  class (R0 instant, R1 refinable with the ancestor it `refines`, R2 deferred, R3 gated) and a stability; `Policy`
  reproduces `stream_policy.executed_sequence` on random trajectories (tests). `none` on a prefix keeps listening;
  `none` at the end sets `handoff`. `undo()` retracts; `Policy(retract=True)` retracts early actions the final decision
  does not confirm; `Policy(residual=True)` reads what follows an early action with a prior line.
  `Policy.replay()` re-scores logged decisions with another threshold.

