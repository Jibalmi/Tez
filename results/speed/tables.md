## Exp 1: many questions about one state (Gemma 4 12B Q8_0 letters unless stated; Laya's protocol, new ticket per call)
| model, server / engine | arm | 1 q p50 / p95 (ms) | 5 q | 10 q | 50 q | ms per q at 50 (p50) | tokens evaluated per 50-q call | GPU SM clock (busy p50 per size) | VRAM guard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12B, production llama-server (-np 1, --cache-ram default), distinct questions | tez serve, today's layout (runtime as deployed) | 155 / 178 | 950 / 1,025 | 1,786 / 1,877 | 10,511 / 12,228 | 210.2 | 0 | 592-825 MHz | clean |
| 12B, production llama-server (-np 1, --cache-ram default), distinct questions | direct /completion per question, today's layout | 129 / 156 | 912 / 1,000 | 1,678 / 1,781 | 10,722 / 11,630 | 214.4 | 8,250 | 592-825 MHz | clean |
| 12B, production llama-server (-np 1, --cache-ram default), distinct questions | direct /completion per question, state first (state cached once per call) | 158 / 263 | 495 / 535 | 752 / 827 | 4,255 / 4,562 | 85.1 | 2,744 | 592-825 MHz | clean |
| 12B, llama-server --cache-ram 0, distinct questions | tez serve, today's layout (runtime as deployed) | 170 / 225 | 1,194 / 1,406 | 2,694 / 3,589 | 16,624 / 17,330 | 332.5 | 0 | 558-648 MHz | clean |
| 12B, llama-server --cache-ram 0, distinct questions | direct /completion per question, today's layout | 132 / 153 | 1,179 / 1,470 | 2,782 / 3,322 | 13,496 / 14,448 | 269.9 | 11,241 | 558-648 MHz | clean |
| 12B, llama-server --cache-ram 0, distinct questions | direct /completion per question, state first (state cached once per call) | 154 / 185 | 522 / 583 | 925 / 1,098 | 4,478 / 4,639 | 89.6 | 2,744 | 558-648 MHz | clean |
| 12B, llama-server -np 8 -kvu --cache-ram 0, distinct questions | concurrent /completion (one per question), today's layout | 93 / 146 | 776 / 831 | 1,757 / 1,840 | 8,044 / 8,176 | 160.9 | 11,106 | 727-997 MHz | clean |
| 12B, llama-server -np 8 -kvu --cache-ram 0, distinct questions | concurrent /completion, state first | 105 / 114 | 709 / 757 | 1,479 / 1,639 | 2,929 / 3,202 | 58.6 | 3,632 | 727-997 MHz | clean |
| 12B, llama-server -np 8 -kvu --cache-ram 0, distinct questions | one /completion with all prompts, state first | 105 / 162 | 704 / 792 | 1,298 / 1,436 | 2,553 / 2,733 | 51.0 | 3,172 | 727-997 MHz | clean |
| 12B, llama-server -np 8 -kvu --cache-ram 0, distinct questions | direct /completion per question, state first (state cached once per call) | 103 / 111 | 392 / 413 | 839 / 876 | 3,951 / 4,026 | 79.0 | 2,700 | 727-997 MHz | clean |
| 12B, in-process llama.dll, distinct questions | sequential, today's layout (KV rollback reuse) | 121 / 125 | 519 / 523 | 1,201 / 1,323 | 6,430 / 6,607 | 128.6 | 11,283 | 877-1324 MHz | clean |
| 12B, in-process llama.dll, distinct questions | sequential, state first (KV rollback reuse) | 95 / 98 | 275 / 286 | 536 / 559 | 2,759 / 3,127 | 55.2 | 2,744 | 877-1324 MHz | clean |
| 12B, in-process llama.dll, distinct questions | state once + all question suffixes in ONE decode (seq_cp) | 120 / 123 | 187 / 191 | 289 / 331 | 1,357 / 1,439 | 27.1 | 2,756 | 877-1324 MHz | clean |
| 12B, in-process llama.dll, distinct questions | one sequence, all questions, read at markers | 96 / 101 | 180 / 184 | 296 / 400 | 1,402 / 1,428 | 28.1 | 2,876 | 877-1324 MHz | clean |
| 12B, production llama-server, Laya's verbatim protocol | tez serve, today's layout (runtime as deployed) | 153 / 207 | 632 / 685 | 1,199 / 1,240 | 5,880 / 5,971 | 117.6 | 0 | 652-787 MHz | clean |
| 12B, production llama-server, Laya's verbatim protocol | direct /completion per question, today's layout | 126 / 165 | 591 / 702 | 1,205 / 1,366 | 5,343 / 6,986 | 106.9 | 378 | 652-787 MHz | clean |
| 12B, production llama-server, Laya's verbatim protocol | direct /completion per question, state first (state cached once per call) | 152 / 207 | 437 / 526 | 856 / 994 | 4,117 / 4,211 | 82.3 | 2,453 | 652-787 MHz | clean |
| 12B, llama-server --cache-ram 0, Laya's verbatim protocol | tez serve, today's layout (runtime as deployed) | 172 / 190 | 1,131 / 1,346 | 2,367 / 2,461 | 11,826 / 12,867 | 236.5 | 0 | 420-645 MHz | clean |
| 12B, llama-server --cache-ram 0, Laya's verbatim protocol | direct /completion per question, state first (state cached once per call) | 168 / 189 | 500 / 538 | 884 / 906 | 4,004 / 4,104 | 80.1 | 2,453 | 420-645 MHz | clean |
| 12B, in-process llama.dll, Laya's verbatim protocol | sequential, today's layout (KV rollback reuse) | 97 / 101 | 486 / 498 | 1,108 / 1,228 | 5,732 / 5,822 | 114.7 | 10,992 | 945-1500 MHz | clean |
| 12B, in-process llama.dll, Laya's verbatim protocol | state once + all question suffixes in ONE decode (seq_cp) | 96 / 102 | 183 / 195 | 274 / 284 | 1,114 / 1,172 | 22.3 | 2,465 | 945-1500 MHz | clean |
| Qwen3.5-4B 24 blocks, in-process, distinct questions | one batch, today's layout (no sharing) | 41 / 44 | 203 / 214 | 393 / 398 | 2,153 / 2,253 | 43.1 | 11,282 | 675-1620 MHz | clean |
| Qwen3.5-4B 24 blocks, in-process, distinct questions | state once + all question suffixes in ONE decode (seq_cp) | 42 / 44 | 94 / 101 | 143 / 149 | 647 / 651 | 12.9 | 2,755 | 675-1620 MHz | clean |

### Typed-decisions accuracy by layout (400 rows x 5 questions)
| layout / engine | accuracy (2,000) | choice | noul | score | prompt tokens evaluated | paired vs HTTP today's layout |
|---|---:|---:|---:|---:|---:|---:|
| today's layout (options first, state last), HTTP | 0.7050 | 0.675 | 0.8133 | 0.6462 | 480,400 | - |
| state first, HTTP, prompt cache | 0.7015 | 0.6583 | 0.8233 | 0.6425 | 246,784 | - |
| 12b_inproc state once + all question suffixes in ONE decode (seq_cp) | 0.7020 | 0.6583 | 0.825 | 0.6425 | - | td_rows_prod_statefirst_http.jsonl: agree 0.989, McNemar p 1.0; td_rows_prod_today_http.jsonl: agree 0.8205, McNemar p 0.77648 |
| 12b_inproc one batch, today's layout (no sharing) | 0.7010 | 0.6767 | 0.8117 | 0.6362 | - | td_rows_prod_statefirst_http.jsonl: agree 0.8125, McNemar p 1.0; td_rows_prod_today_http.jsonl: agree 0.9825, McNemar p 0.18493 |
| 12b_inproc one sequence, all questions, read at markers | 0.3345 | 0.2267 | 0.5867 | 0.2263 | - | td_rows_prod_statefirst_http.jsonl: agree 0.3695, McNemar p 0.0; td_rows_prod_today_http.jsonl: agree 0.3855, McNemar p 0.0 |
| q4bL24_inproc state once + all question suffixes in ONE decode (seq_cp) | 0.5740 | 0.5983 | 0.675 | 0.48 | - | - |
| q4bL24_inproc one batch, today's layout (no sharing) | 0.5815 | 0.5433 | 0.715 | 0.51 | - | - |

## Exp 2: probes (Qwen3.5-4B cut to 24 blocks)
### Accuracy
| features (Qwen3.5-4B, 24 blocks) | probe accuracy (2,000) | choice | noul | score | C |
|---|---:|---:|---:|---:|---:|
| state only, one vector per state (llama-server /embedding) | 0.7490 | 0.7183 | 0.8367 | 0.7063 | 0.05 |
| state only, one vector per state (in-process) | 0.7445 | 0.7083 | 0.8367 | 0.7025 | 0.05 |
| state first, then the question: one vector per question, state shared (in-process) | 0.7845 | 0.7433 | 0.85 | 0.7662 | 0.05 |
| question first, state last: one prompt per question (served, BENCHMARKS) | 0.7930 | 0.7667 | 0.8717 | 0.7538 | 0.05 |

### Latency per call (Laya's protocol)
| engine | layout | 1 q p50 / p95 (ms) | 2 q | 5 q | 10 q | 20 q | 50 q | ms per q at 50 | VRAM guard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama-server /embedding (-np 1) | stateonly | 40.9 / 43.4 | 40.8 / 43.5 | 40.9 / 45.2 | 40.3 / 43.0 | 41.4 / 44.0 | 41.2 / 45.3 | 0.82 | flagged by the 200 MB rule: own pinned host output buffer (582 MB) of the /embedding path; 5.5 of 15.9 GB VRAM in use, no spill |
| llama-server -np 8, one /embedding per question, concurrent | qprompt | 49.1 / 54.0 | 123.1 / 131.0 | 304.5 / 314.3 | 618.0 / 636.5 | 1,262.4 / 1,318.6 | 3,750.4 / 4,018.6 | 75.01 | flagged by the 200 MB rule: own pinned host output buffer (586 MB) of the /embedding path; 5.9 of 15.9 GB VRAM in use, no spill |
| llama-server -np 8, one /embedding with all prompts | qprompt | 51.2 / 56.6 | 133.3 / 141.3 | 320.6 / 329.1 | 666.4 / 704.3 | 1,555.4 / 1,763.2 | 5,085.9 / 5,331.2 | 101.72 | flagged by the 200 MB rule: own pinned host output buffer (586 MB) of the /embedding path; 6.0 of 15.9 GB VRAM in use, no spill |
| in-process llama.dll | stateonly | 40.4 / 68.1 | 36.4 / 43.6 | 36.2 / 40.4 | 38.3 / 50.6 | 36.6 / 41.5 | 37.8 / 44.3 | 0.76 | clean |
| in-process llama.dll | sfq | 45.8 / 81.1 | 135.0 / 221.1 | 212.0 / 245.4 | 346.4 / 367.7 | 657.8 / 712.4 | 2,103.9 / 2,463.1 | 42.08 | clean |
| in-process llama.dll | qprompt | 45.1 / 50.0 | 125.0 / 212.4 | 419.2 / 443.6 | 1,080.4 / 1,183.4 | 2,459.7 / 3,294.9 | 6,781.2 / 8,334.9 | 135.62 | clean |

## Exp 3: streamed voice word -> action (220 commands, 1,158 words; incremental words k >= 2)
| model / engine | compute ms per word p50 / p95 | round trip ms per word p50 / p95 | tokens per word | intent acc. | acc. accepting then/alt | none recall / precision | harmful | OOS false actions | first action word | first action <= human word | thermal-throttled share | VRAM guard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 12B letters, production llama-server (cache on, the published baseline) | 33.5 / 42.8 | 41.6 / 51.6 | 11 | 0.909 | 0.982 | 1.00 / 0.92 | 1 / 198 | 2 / 22 | 3.62 | 0.53 | 0.239 | clean |
| Gemma 4 12B letters, in-process | 34.4 / 38.8 | 35.6 / 40.5 | 13 | 0.909 | 0.986 | 1.00 / 0.92 | 1 / 198 | 2 / 22 | 3.62 | 0.53 | 0.0 | clean |
| Qwen3.5-4B (32 blocks) letters, in-process | 31.8 / 42.3 | 33.0 / 44.0 | 13 | 0.818 | 0.918 | 0.91 / 0.65 | 5 / 198 | 3 / 22 | 4.40 | 0.39 | 0.0 | clean |
| Qwen3.5-4B 24 blocks letters, llama-server, cache off (the runtime's Qwen setting) | 68.0 / 73.5 | 74.9 / 81.2 | 356 | 0.605 | 0.654 | 0.73 / 0.94 | 80 / 198 | 7 / 22 | 4.20 | 0.45 | 0.0 | clean |
| Qwen3.5-4B 24 blocks letters, in-process | 24.9 / 32.7 | 26.0 / 34.1 | 13 | 0.618 | 0.673 | 0.77 / 0.94 | 79 / 198 | 6 / 22 | 4.10 | 0.46 | 0.0 | clean |
| Qwen3.5-4B 24 blocks probe at the last word (2-fold CV), llama-server /embedding, cache on (exact extension) [full] | - / - | 19.6 / 24.8 | - | 0.541 | 0.609 | 0.59 / 0.59 | 56 / 198 | 10 / 22 | 4.44 | 0.46 | 0.0 | flagged by the 200 MB rule: own pinned host output buffer (422 MB) of the /embedding path; 5.6 of 15.9 GB VRAM in use, no spill |
| Qwen3.5-4B 24 blocks probe at the last word (2-fold CV), llama-server /embedding, cache on (exact extension) [prefix] | - / - | 19.6 / 24.8 | - | 0.586 | 0.664 | 0.68 / 0.41 | 37 / 198 | 8 / 22 | 3.93 | 0.46 | 0.0 | flagged by the 200 MB rule: own pinned host output buffer (422 MB) of the /embedding path; 5.6 of 15.9 GB VRAM in use, no spill |
| Qwen3.5-4B 24 blocks probe at the last word (2-fold CV), in-process, exact extension [full] | 9.9 / 14.9 | 10.9 / 16.2 | 1 | 0.532 | 0.600 | 0.59 / 0.57 | 57 / 198 | 11 / 22 | 4.30 | 0.48 | 0.0 | clean |
| Qwen3.5-4B 24 blocks probe at the last word (2-fold CV), in-process, exact extension [prefix] | 9.9 / 14.9 | 10.9 / 16.2 | 1 | 0.573 | 0.654 | 0.68 / 0.38 | 40 / 198 | 8 / 22 | 3.97 | 0.44 | 0.0 | clean |
| Qwen3.5-4B 24 blocks probe at the answer position (2-fold CV), in-process [full] | 29.0 / 36.3 | 30.1 / 37.8 | 13 | 0.900 | 0.918 | 0.86 / 0.79 | 7 / 198 | 4 / 22 | 4.02 | 0.49 | 0.0 | clean |
| Qwen3.5-4B 24 blocks probe at the answer position (2-fold CV), in-process [prefix] | 29.0 / 36.3 | 30.1 / 37.8 | 13 | 0.882 | 0.891 | 0.82 / 0.50 | 3 / 198 | 4 / 22 | 3.89 | 0.42 | 0.0 | clean |
| Qwen3.5-4B (32 blocks) probe at the answer position (2-fold CV), in-process [full] | 38.8 / 47.5 | 40.1 / 49.3 | 13 | 0.891 | 0.918 | 0.86 / 0.76 | 7 / 198 | 4 / 22 | 4.00 | 0.48 | 0.0 | clean |
| Qwen3.5-4B (32 blocks) probe at the answer position (2-fold CV), in-process [prefix] | 38.8 / 47.5 | 40.1 / 49.3 | 13 | 0.891 | 0.900 | 0.86 / 0.53 | 3 / 198 | 3 / 22 | 3.92 | 0.41 | 0.0 | clean |
| Qwen3.5-4B 24 blocks probe at the answer position (2-fold CV), in-process, deferred commit [full] | 18.2 / 22.9 | 19.6 / 24.4 | 11 | 0.900 | 0.918 | 0.86 / 0.79 | 10 / 198 | 4 / 22 | 3.95 | 0.51 | 0.0 | clean |
| Qwen3.5-4B 24 blocks probe at the answer position (2-fold CV), in-process, deferred commit [prefix] | 18.2 / 22.9 | 19.6 / 24.4 | 11 | 0.882 | 0.891 | 0.82 / 0.51 | 3 / 198 | 4 / 22 | 3.87 | 0.43 | 0.0 | clean |
| Qwen3.5-4B 24 blocks probe after '"<|im_end|>' (2-fold CV), in-process, deferred commit [full] | 14.7 / 18.6 | 15.9 / 19.8 | 3 | 0.768 | 0.782 | 0.91 / 0.80 | 41 / 198 | 2 / 22 | 4.52 | 0.37 | 0.0 | clean |
| Qwen3.5-4B 24 blocks probe after '"<|im_end|>' (2-fold CV), in-process, deferred commit [prefix] | 14.7 / 18.6 | 15.9 / 19.8 | 3 | 0.768 | 0.773 | 0.77 / 0.47 | 26 / 198 | 5 / 22 | 4.16 | 0.39 | 0.0 | clean |
| Gemma 4 12B letters, in-process, deferred commit | 36.9 / 47.6 | 38.7 / 50.3 | 11 | 0.909 | 0.986 | 1.00 / 0.92 | 1 / 198 | 2 / 22 | 3.62 | 0.53 | 0.793 | clean |

## Exp 4: single-question overhead
| server | path | round trip p50 / p95 (ms) | llama prompt_ms p50 | tokens evaluated | response bytes | VRAM guard |
|---|---:|---:|---:|---:|---:|---:|
| production llama-server | client -> tez serve -> llama-server (runtime) | 144.8 / 195.2 | - | - | 362 | clean |
| production llama-server | Tez engine in-process -> llama-server | 145.1 / 174.7 | - | - | - | clean |
| production llama-server | direct /completion, n_probs 200 | 140.0 / 197.4 | 81.4 | 164 | 18,605 | clean |
| production llama-server | direct /completion, n_probs 20 | 146.8 / 194.6 | 87.0 | 164 | 4,178 | clean |
| production llama-server | direct /completion, n_probs 0 | 140.1 / 208.3 | 86.7 | 164 | 2,668 | clean |
| production llama-server | GET /health (HTTP floor) | 1.0 / 1.4 | - | - | - | clean |
| llama-server --cache-ram 0 | client -> tez serve -> llama-server (runtime) | 149.0 / 162.3 | - | - | 361 | clean |
| llama-server --cache-ram 0 | Tez engine in-process -> llama-server | 148.4 / 163.9 | - | - | - | clean |
| llama-server --cache-ram 0 | direct /completion, n_probs 200 | 147.7 / 271.4 | 138.2 | 164 | 18,604 | clean |
| llama-server --cache-ram 0 | direct /completion, n_probs 20 | 144.2 / 166.6 | 137.7 | 164 | 4,180 | clean |
| llama-server --cache-ram 0 | direct /completion, n_probs 0 | 143.7 / 165.0 | 140.9 | 164 | 2,668 | clean |
| llama-server --cache-ram 0 | GET /health (HTTP floor) | 0.9 / 1.3 | - | - | - | clean |

## Qwen3.5 prompt-cache safety on llama-server b11100
| model | scenario | server | steps (evaluated / reused tokens, ms) |
|---|---:|---:|---:|
| L24 | identical | ok | /completion n=354 cached=0 239.2 ms; /completion n=4 cached=350 66.3 ms |
| L24 | extend_n0 | ok | /completion n=343 cached=0 127.1 ms; /completion n=4 cached=343 51.8 ms; /completion n=1 cached=347 49.2 ms |
| L24 | extend_n1 | ok | /completion n=343 cached=0 132.5 ms; /completion n=4 cached=343 42.2 ms |
| L24 | voice_letters | ok | /completion n=353 cached=0 180.1 ms; /completion n=354 cached=0 111.3 ms; /completion n=355 cached=0 108.9 ms |
| L24 | embed_extend | ok | /embedding n=None cached=None 264.0 ms; /embedding n=None cached=None 27.4 ms; /embedding n=None cached=None 23.0 ms |
| L24 | diverge | ok | /completion n=221 cached=0 91.5 ms; /completion n=186 cached=0 100.8 ms |
| L24 | multiturn_checkpoint | ok | /completion n=157 cached=0 96.2 ms; /completion n=4 cached=153 49.8 ms; /completion n=152 cached=0 95.9 ms |
| L32 | identical | ok | /completion n=354 cached=0 158.3 ms; /completion n=4 cached=350 79.6 ms |
| L32 | extend_n0 | ok | /completion n=343 cached=0 280.0 ms; /completion n=4 cached=343 53.4 ms; /completion n=1 cached=347 78.3 ms |
| L32 | extend_n1 | ok | /completion n=343 cached=0 302.8 ms; /completion n=4 cached=343 84.3 ms |
| L32 | voice_letters | ok | /completion n=353 cached=0 153.6 ms; /completion n=354 cached=0 137.5 ms; /completion n=355 cached=0 137.3 ms |
| L32 | embed_extend | ok | /embedding n=None cached=None 218.5 ms; /embedding n=None cached=None 27.7 ms; /embedding n=None cached=None 28.2 ms |
| L32 | diverge | ok | /completion n=221 cached=0 277.4 ms; /completion n=186 cached=0 152.3 ms |
| L32 | multiturn_checkpoint | ok | /completion n=157 cached=0 95.5 ms; /completion n=4 cached=153 56.6 ms; /completion n=152 cached=0 98.5 ms |
