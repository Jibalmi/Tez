"""Author rule-supervised records without reading benchmark examples or diagnostic templates.

Domain-specific field names are disjoint across training, validation and confirmation. The generic
ranking schema, choice labels and logical operations recur across splits; the overlap filter must
screen them before training. This is not held-out logic-family transfer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/pilot-reasoning-data-20260927"
DOMAINS = {
    "train": [
        ("library allocations", "borrowed_items", "borrowing_cap", "membership_checked", "hold_present"),
        ("bicycle rentals", "active_rentals", "rental_cap", "lock_checked", "service_hold"),
        ("event badges", "issued_badges", "badge_allotment", "desk_checked", "print_hold"),
        ("parcel lockers", "occupied_lockers", "locker_capacity", "label_checked", "storage_hold"),
        ("maker reservations", "reserved_sessions", "session_cap", "induction_checked", "workshop_hold"),
        ("media downloads", "used_downloads", "download_allowance", "account_checked", "transfer_hold"),
    ],
    "validation": [
        ("repair intake", "waiting_repairs", "repair_capacity", "inspection_checked", "parts_hold"),
        ("supply inventory", "requested_packs", "pack_allotment", "stock_checked", "shipment_hold"),
    ],
    "confirmation": [
        ("community swimming", "admitted_swimmers", "admission_cap", "entry_checked", "pool_hold"),
        ("music lending", "loaned_instruments", "loan_allowance", "return_checked", "tuning_hold"),
    ],
}
PER_DOMAIN = {"train": 300, "validation": 120, "confirmation": 180}
RACKS = ("rack_elm", "rack_fir", "rack_oak")


def labels(facts, racks, negated):
    above = facts["usage"] > facts["cap"]
    permitted = facts["checked"] and not above
    return {
        "compare": "true" if above else "false",
        "combined": "true" if (not permitted if negated else permitted) else "false",
        "policy": "pause_cycle" if facts["held"] else "buffer_cycle" if above else "dispatch_cycle",
        "count": str(int(above) + int(facts["held"])),
        "largest": max(RACKS, key=lambda key: racks[key]),
    }


def make_record(domain, index, rng, split):
    name, used, cap, checked, held = domain
    count = index % 3
    above = count == 2 or (count == 1 and bool((index // 3) % 2))
    hold = count == 2 or (count == 1 and not above)
    ceiling = rng.randrange(3, 66)
    usage = ceiling + rng.choice((1, 2, 7)) if above else ceiling - rng.choice((0, 1, 2))
    verified = bool((index // 6) % 2)
    negate = bool((index // 12) % 2)
    facts = {"usage": usage, "cap": ceiling, "checked": verified, "held": hold}
    racks = {key: rng.randrange(33) for key in RACKS}
    state = {"scenario": name, "record": {used: usage, cap: ceiling, checked: verified, held: hold}, "racks": racks}
    targets = labels(facts, racks, negate)
    combined = (f"Does at least one condition fail: {checked} is true; {used} is no greater than {cap}?" if negate else
                f"Do both conditions hold: {checked} is true AND {used} is no greater than {cap}?")
    definitions = [
        ("compare", "noul", f"Does {used} exceed {cap}? Equality does not count as exceeding.", [("true", "Yes"), ("false", "No")]),
        ("combined", "noul", combined, [("true", "Yes"), ("false", "No")]),
        ("policy", "choice", f"Apply these rules in priority order: if {held} is true, pause the cycle; "
         f"otherwise if {used} exceeds {cap}, buffer the cycle; otherwise dispatch the cycle.",
         [("dispatch_cycle", "Dispatch the cycle"), ("buffer_cycle", "Buffer the cycle"), ("pause_cycle", "Pause the cycle")]),
        ("count", "score", f"Count how many of these two conditions hold: {used} exceeds {cap}; {held} is true.",
         [("0", "Neither condition holds"), ("1", "Exactly one condition holds"), ("2", "Both conditions hold")]),
        ("largest", "choice", "Which rack has the largest number? Break ties in this priority order: rack_elm, rack_fir, rack_oak.",
         [(key, f"Select {key}") for key in RACKS]),
    ]
    questions = []
    for operation, qtype, instructions, options in definitions:
        questions.append({"operation": operation, "qtype": qtype, "instructions": instructions, "options": options,
                          "gold_key": targets[operation]})
    return {"id": f"reasoning-v1/{split}/{name.replace(' ', '-')}/{index:04d}", "domain": name, "split": split,
            "group": f"{name}/{index // 24:03d}", "state": state, "questions": questions,
            "oracle_inputs": {"facts": facts, "racks": racks, "negated": negate},
            "field_mapping": {"usage": used, "cap": cap, "checked": checked, "held": held}}


def generate():
    output = {}
    for split, domains in DOMAINS.items():
        rows = []
        for domain in domains:
            seed = int(hashlib.sha256(("fresh-rule-v1/" + split + "/" + domain[0]).encode()).hexdigest()[:16], 16)
            rng = random.Random(seed)
            rows.extend(make_record(domain, i, rng, split) for i in range(PER_DOMAIN[split]))
        random.Random(713).shuffle(rows)
        output[split] = rows
    return output


def validate(data):
    ids, states, domains, schemas = set(), set(), {}, {}
    for split, rows in data.items():
        domains[split], schemas[split] = set(), set()
        for record in rows:
            if record["split"] != split or record["id"] in ids:
                raise ValueError("Split or identifier mismatch")
            ids.add(record["id"])
            state_key = json.dumps(record["state"], sort_keys=True)
            if state_key in states:
                raise ValueError("Duplicate state across the authored corpus")
            states.add(state_key)
            facts = {key: record["state"]["record"][field] for key, field in record["field_mapping"].items()}
            if facts != record["oracle_inputs"]["facts"] or record["state"]["racks"] != record["oracle_inputs"]["racks"]:
                raise ValueError("Oracle metadata differs from model-visible facts")
            expected = labels(**record["oracle_inputs"])
            for q in record["questions"]:
                if q["gold_key"] != expected[q["operation"]] or q["gold_key"] not in dict(q["options"]):
                    raise ValueError("Oracle label mismatch")
                # The shared ranking schema is explicitly exempt from the domain-specific schema claim.
                if q["operation"] != "largest":
                    schemas[split].add((q["qtype"], q["instructions"]))
            domains[split].add(record["domain"])
    keys = list(data)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            if domains[left] & domains[right] or schemas[left] & schemas[right]:
                raise ValueError("Domain-specific schema overlap across splits")


def prepare(directory):
    data = generate()
    validate(data)
    directory.mkdir(parents=True, exist_ok=False)
    for split, rows in data.items():
        write_json(directory / f"{split}.json", rows)
    manifest = {"version": "reasoning-v1", "generator_sha256": digest(__file__),
                "source": "Deterministically authored fixtures; no benchmark, probe or teacher file is read by the generator.",
                "supervision": "Exact rule-derived labels; not model probabilities or teacher logits.",
                "status": "unfiltered_do_not_train", "seed_scheme": "SHA256 of fresh-rule-v1/split/domain, first 64 bits; shuffle seed 713",
                "split_policy": "Disjoint domains and field-name-specific schemas. The generic largest-rack schema and all five logical operations recur across splits; do not claim logic-family or complete schema holdout.",
                "training_permission": "Training loader must require a completed leakage-filter receipt. Validation and confirmation never enter training.",
                "splits": {split: {"states": len(rows), "decisions": sum(len(r["questions"]) for r in rows),
                                    "sha256": digest(directory / f"{split}.json"),
                                    "domains": sorted({r["domain"] for r in rows}),
                                    "semantic_label_counts": {op: dict(Counter(q["gold_key"] for r in rows for q in r["questions"]
                                                                               if q["operation"] == op))
                                                              for op in ("compare", "combined", "policy", "count", "largest")}}
                           for split, rows in data.items()}}
    write_json(directory / "manifest.json", manifest)
    print(json.dumps({split: {k: v[k] for k in ("states", "decisions", "sha256")} for split, v in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.output)
