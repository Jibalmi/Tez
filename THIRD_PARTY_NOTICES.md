# Third-party notices

Tez's own code is under the MIT licence (`LICENSE`). This file lists the third-party material in this repository, where
it lives, and the terms it comes under.

## SemIf fixtures and prompt code (MIT)

- Where: `data/semif/` (the public fixtures, `core.py` prompt construction and evaluation scripts), vendored.
- Copyright (c) 2026 TheoLeeCJ, from <https://github.com/TheoLeeCJ/SemIf>.
- Licence: MIT. The full text is in `data/semif/LICENSE`.

## Geist and Geist Mono (SIL Open Font License 1.1)

- Where: `site/assets/fonts/Geist-Variable.woff2` and `site/assets/fonts/GeistMono-Variable.woff2`, self-hosted by the
  website and distributed unmodified.
- Copyright (c) 2023 Vercel, in collaboration with basement.studio.
- Licence: SIL Open Font License, Version 1.1. The full text is in `site/assets/fonts/LICENSE-Geist.txt`. The fonts may
  be bundled with software; they may not be sold by themselves.

## Lucide icons (ISC)

- Where: the website's interface icons, drawn as inline SVG paths in `site/*.html` and `site/assets/js/tez-ui.js`.
- Licence: ISC.

```
ISC License

Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part of Feather (MIT). All other copyright (c) for Lucide are held by Lucide Contributors 2022.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
```

## Code adapted from Laya (Apache-2.0)

Laya is the open encoder-based "System One" that the head-to-head benchmarks compare against
(<https://github.com/NandhaKishorM/laya>, copyright (c) Convai Innovations). Its code is licensed under the Apache
License, Version 2.0; the full text is in [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt). Laya v0.3.20
(commit 23a1752) has no `NOTICE` file, so there are no further notices to carry (Apache-2.0 section 4(d)).

Every adapted file starts with this attribution and a note of what was changed (Apache-2.0 section 4(b)):

```python
# Adapted from laya v0.3.20 @23a1752, (c) Convai Innovations, Apache-2.0; modified.
# Changes: <what was changed>.
```

| File | Adapted from | Changes |
|---|---|---|
| `tez/state.py` | `laya/email.py` | `clean_email_body` renamed `clean_email`; the `email_questions` re-export dropped; docstrings rewritten; type hints modernised. The cleaning rules and their order are Laya's. |
| `tests/test_state.py` | `tests/test_email.py` | Rewritten as pytest parametrisations against `tez.state`; the `email_questions` cases dropped; a test that decisions read the cleaned body added. |

`tez/state.py` is in the wheel, so the wheel carries `LICENSES/Apache-2.0.txt` and this file next to `LICENSE`
(`license-files` in `pyproject.toml`).

Laya's hooks, presets, MCP server, request limits and batch prediction informed the design of Tez's own versions
(`tez/hooks.py`, `tez/presets/`, `tez/mcp_server.py`, `tez/config.py`, the batch endpoint); no code was copied from
them.

## Not distributed with Tez

- **Model weights.** Gemma 4 and Qwen3.5 are downloaded by the user (Apache-2.0); no weights are in the repository or
  in any package or image built from it.
- **llama.cpp** (MIT) runs as a separate program, or as its own official container image in `compose.yaml`; it is not
  bundled.
- **Dependencies** are installed by pip or npm under their own licences. The TypeScript client (`clients/ts`) has no
  runtime dependencies; TypeScript (Apache-2.0) is used only to build it.
