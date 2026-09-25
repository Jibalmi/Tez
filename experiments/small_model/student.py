"""Option A's student: an mmBERT encoder with dev-0.4b's option-span head (plan §4).

A row is the state, then 1-8 questions. Each question is a question marker (mmBERT's <unused0>) followed by
"(type) instructions", then each option as an option marker (<unused1>) followed by "label: description" (the same
option text Tez's letter prompts show). The head scores an option with Linear(2H, H) - GELU - Linear(H, 1) over
[the question marker's state; the mean of the option's tokens (marker included)]; a softmax runs over the options of
each question. The anchor is the question marker, not [CLS], because a packed row holds several questions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import common  # noqa: F401  (puts the repo root on sys.path for tez)

BOS, EOS, PAD = 2, 1, 0
Q_MARK, O_MARK = 7, 8                  # mmBERT's <unused0>, <unused1>
MAX_OPT_TOKENS = 48
MAX_HEAD_TOKENS = 160


def option_text(key: str, desc: Any) -> str:
    """As tez.prompt.render_option_lines shows an option, without the letter."""
    shown = "none" if key == "__none__" else str(key)
    if isinstance(desc, str) and desc and desc != shown:
        return f"{shown}: {desc}"
    return shown


def render_state(state: Any) -> str:
    from tez.prompt import render_state as rs
    return rs(state)


@dataclass
class Q:
    """One question as the student sees it: its type, instructions and options [(key, description)]."""
    type: str
    instructions: str
    options: list


@dataclass
class Row:
    ids: list[int]
    q_pos: list[int]                                  # position of each question marker
    opt_spans: list[list[tuple[int, int]]]            # per question, per option: [start, end) token span
    meta: dict = field(default_factory=dict)


class Encoder:
    """Turns (state, questions) into token rows; tokenises pieces separately so every span is exact."""

    def __init__(self, tokenizer, max_len: int = 2048):
        self.tok = tokenizer
        self.max_len = max_len
        self._cache: dict[str, list[int]] = {}

    def ids(self, text: str) -> list[int]:
        v = self._cache.get(text)
        if v is None:
            v = self.tok(text, add_special_tokens=False)["input_ids"]
            if len(self._cache) < 200000:
                self._cache[text] = v
        return v

    def encode(self, state: Any, questions: Sequence[Q], state_text: str | None = None) -> Row:
        parts = []
        q_len = 0
        for q in questions:
            head = self.ids(f" ({q.type}) {q.instructions}")[:MAX_HEAD_TOKENS]
            opts = [self.ids(" " + option_text(k, d))[:MAX_OPT_TOKENS] for k, d in q.options]
            parts.append((head, opts))
            q_len += 1 + len(head) + sum(1 + len(o) for o in opts)
        st = self.ids(state_text if state_text is not None else render_state(state))
        budget = max(32, self.max_len - q_len - 3)
        st = st[:budget]
        ids = [BOS] + st + [EOS]
        q_pos, spans = [], []
        for head, opts in parts:
            q_pos.append(len(ids))
            ids.append(Q_MARK)
            ids.extend(head)
            sp = []
            for o in opts:
                s = len(ids)
                ids.append(O_MARK)
                ids.extend(o)
                sp.append((s, len(ids)))
            spans.append(sp)
        ids.append(EOS)
        return Row(ids, q_pos, spans)


def collate(rows: Sequence[Row], device) -> dict:
    """Pad a batch; flatten the questions and options into index tensors for the head."""
    L = max(len(r.ids) for r in rows)
    B = len(rows)
    input_ids = torch.full((B, L), PAD, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    qb, qp = [], []                    # question -> (batch row, marker position)
    tok_b, tok_p, tok_o = [], [], []   # option token -> (row, position, option id)
    opt_q, opt_len = [], []            # option -> question id, token count
    q_opts = []                        # question -> list of option ids
    for b, r in enumerate(rows):
        input_ids[b, : len(r.ids)] = torch.tensor(r.ids)
        attn[b, : len(r.ids)] = 1
        for qi, (pos, spans) in enumerate(zip(r.q_pos, r.opt_spans)):
            qid = len(qb)
            qb.append(b)
            qp.append(pos)
            ids_here = []
            for s, e in spans:
                oid = len(opt_q)
                opt_q.append(qid)
                opt_len.append(e - s)
                ids_here.append(oid)
                tok_b.extend([b] * (e - s))
                tok_p.extend(range(s, e))
                tok_o.extend([oid] * (e - s))
            q_opts.append(ids_here)
    K = max(len(x) for x in q_opts)
    q_opt_index = torch.full((len(q_opts), K), -1, dtype=torch.long)
    for i, x in enumerate(q_opts):
        q_opt_index[i, : len(x)] = torch.tensor(x)
    t = lambda v: torch.tensor(v, dtype=torch.long, device=device)  # noqa: E731
    return {"input_ids": input_ids.to(device), "attention_mask": attn.to(device), "q_b": t(qb), "q_p": t(qp),
            "tok_b": t(tok_b), "tok_p": t(tok_p), "tok_o": t(tok_o), "opt_q": t(opt_q),
            "opt_len": torch.tensor(opt_len, dtype=torch.float32, device=device), "q_opt_index": q_opt_index.to(device)}


class Student(nn.Module):
    def __init__(self, name_or_path: str, dtype=torch.float32):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(name_or_path, dtype=dtype, attn_implementation="sdpa")
        H = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.Linear(2 * H, H), nn.GELU(), nn.Linear(H, 1))

    def forward(self, batch: dict) -> torch.Tensor:
        """Log-probabilities per question over its options: (n_questions, K_max), -inf where a question has fewer."""
        h = self.enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        hq = h[batch["q_b"], batch["q_p"]]                                   # (NQ, H)
        ht = h[batch["tok_b"], batch["tok_p"]]                               # (NT, H)
        n_opt = batch["opt_q"].shape[0]
        ho = torch.zeros((n_opt, h.shape[-1]), dtype=ht.dtype, device=h.device).index_add_(0, batch["tok_o"], ht)
        ho = ho / batch["opt_len"].to(ho.dtype).unsqueeze(1)
        s = self.head(torch.cat([hq[batch["opt_q"]], ho], dim=-1).float()).squeeze(-1)   # (NO,)
        idx = batch["q_opt_index"]
        scores = torch.where(idx >= 0, s[idx.clamp(min=0)], torch.full_like(idx, -1e4, dtype=s.dtype))
        return F.log_softmax(scores, dim=-1)


def question_from_wire(qd: dict, abstain: bool = False) -> Q:
    """A Tez wire question (type, instructions, criteria) as the student sees it (options as tez shows them)."""
    from tez.schema import parse_question
    q = parse_question(qd.get("qid", "q"), {k: qd[k] for k in ("type", "instructions", "criteria") if k in qd})
    return Q(q.type, q.instructions_text, q.options(abstain))


def ordinal_loss(logp: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """dev-0.4b's ordinal term: squared distance between the teacher's and the student's cumulative distributions."""
    p = logp.exp() * mask
    return ((torch.cumsum(p, -1) - torch.cumsum(target * mask, -1)) ** 2 * mask).sum(-1)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total - step) / max(1, total - warmup))
    return f


def cosine_sim(a, b):
    return float((a @ b) / (math.sqrt(float(a @ a)) * math.sqrt(float(b @ b)) + 1e-9))
