Timing runs from 2026-09-24 09:19-11:17 that cannot be shown clean: thermal throttling without pacing, and/or an
Ollama runner (gemma4:12b Q4_K_M, 8.2 GB, loaded 10:32 by another session) holding VRAM, and/or VRAM spill of our
llama-server into shared system memory. Kept only as a record; every number here was re-measured under the GPU lock
with before/after VRAM snapshots (gpu_guard in each result JSON). Accuracy numbers in these files are unaffected
(voice_12b_http_prod.json accuracy/policy = the clean rerun's), timings are not results.
