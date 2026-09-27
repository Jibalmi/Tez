# English GLiNER probe: UTF-8 output correction

The [original English attempt](../pilot-gliner2-probe-20260927/README.md) is terminal: native model
construction raised `UnicodeEncodeError` while printing an emoji to CP1252 output. This retry
sets `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` only in the child process. It repeats the same
four non-benchmark interface inputs, with the same checkpoint, FP32 precision and tolerances.

The multilingual checkpoint's completed first-attempt checks remain valid and are not rerun.
This driver selects only English, under the shared compute lock, four threads, unchanged thermal
rules and a three-minute worker deadline. No accuracy, latency or cloud-spending claim follows.

The retry completed all four checks successfully. See the [native output receipt](GLiNER2.5-Decide.json)
for logits, complete probabilities, encoded shapes, checkpoint revision and source hashes.
The UTF-8 correction was sufficient; model precision and case contents were unchanged.
