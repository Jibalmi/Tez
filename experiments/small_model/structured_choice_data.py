"""Author exact choice tasks with disjoint domain vocabularies; no external examples are read."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from structure_probe import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/pilot-structured-choice-20260927"
RESULTS = ROOT / "results/pilot-structured-choice-20260927"
DOMAINS = {
    "train": ("mosaic", "lantern", "origami", "ribbon", "tapestry", "marble"),
    "validation": ("stencil", "bookmark"),
    "confirmation": ("wreath", "costume"),
}
PER_DOMAIN = {"train": 200, "validation": 100, "confirmation": 150}


def definitions(domain, priority):
    fields = {"amount": domain + "_quantity", "limit": domain + "_capacity", "hold": domain + "_reserved"}
    actions = [domain + suffix for suffix in ("_retain", "_stage", "_release")]
    instructions = {
        "policy": f"Apply the first matching rule: if {fields['hold']} is true, choose {actions[0]}; "
                  f"otherwise if {fields['amount']} is greater than {fields['limit']}, choose {actions[1]}; "
                  f"otherwise choose {actions[2]}. Equality is not greater than.",
        "largest": "Select the bin with the greatest stored count. If several share that count, select the first in "
                   "this priority list: " + ", ".join(priority) + ".",
    }
    return fields, actions, instructions


def make_record(split, domain, index, rng):
    entities = [f"{domain}_bin_{letter}" for letter in "abcdefg"[:3 + index % 5]]
    priority = rng.sample(entities, len(entities))
    fields, actions, instructions = definitions(domain, priority)
    branch = index % 3
    held = branch == 0
    above = branch == 1 or (held and bool(rng.randrange(2)))
    limit = rng.randrange(5, 81)
    amount = limit + rng.randrange(1, 9) if above else limit - (0 if index % 4 == 0 else rng.randrange(1, 4))
    bins = {key: rng.randrange(30) for key in entities}
    winner = entities[(index // 5) % len(entities)]
    peak = rng.randrange(30, 101)
    bins[winner] = peak
    if index % 7 == 0:
        bins[rng.choice([key for key in entities if key != winner])] = peak
    # Dict order is irrelevant to the explicit priority list and option order.
    bins = {key: bins[key] for key in rng.sample(entities, len(entities))}
    state = {"workshop": domain, "status": {fields["amount"]: amount, fields["limit"]: limit, fields["hold"]: held}, "bins": bins}
    target = {"policy": actions[0] if held else actions[1] if above else actions[2],
              "largest": max(priority, key=bins.__getitem__)}
    options = {"policy": list(zip(actions, (f"Retain the {domain} items", f"Stage the {domain} items", f"Release the {domain} items"))),
               "largest": [(key, f"Select bin {key}") for key in entities]}
    for value in options.values():
        rng.shuffle(value)
    return {"id": f"structured-choice-v1/{split}/{domain}/{index:04d}", "split": split, "domain": domain,
            "group": f"{domain}/{index // 20:02d}", "state": state, "priority": priority,
            "questions": [{"operation": operation, "qtype": "choice", "instructions": instructions[operation],
                           "options": options[operation], "gold_key": target[operation]} for operation in ("policy", "largest")]}


def generate():
    data = {}
    for split, domains in DOMAINS.items():
        rows = []
        for domain in domains:
            seed = int(hashlib.sha256(f"structured-choice-v1/{split}/{domain}".encode()).hexdigest()[:16], 16)
            rng = random.Random(seed)
            rows.extend(make_record(split, domain, index, rng) for index in range(PER_DOMAIN[split]))
        random.Random(619).shuffle(rows)
        data[split] = rows
    return data


def independent_gold(record):
    """Derive the answers from visible facts using predicate priority and a sorted tie break."""
    fields, actions, _ = definitions(record["domain"], record["priority"])
    status, bins = record["state"]["status"], record["state"]["bins"]
    predicates = (status[fields["hold"]], status[fields["amount"]] > status[fields["limit"]], True)
    policy = next(action for predicate, action in zip(predicates, actions, strict=True) if predicate)
    ranked = sorted(enumerate(record["priority"]), key=lambda item: (-bins[item[1]], item[0]))
    return {"policy": policy, "largest": ranked[0][1]}


def validate(data):
    ids, states = set(), set()
    vocabulary = {}
    for split, rows in data.items():
        vocabulary[split] = set()
        for row in rows:
            if row["split"] != split or row["domain"] not in DOMAINS[split] or row["id"] in ids:
                raise ValueError("Unexpected split/domain membership or duplicate identifier")
            ids.add(row["id"])
            serialized = json.dumps(row["state"], sort_keys=True)
            if serialized in states:
                raise ValueError("Duplicate state")
            states.add(serialized)
            fields, actions, instructions = definitions(row["domain"], row["priority"])
            status, bins = row["state"]["status"], row["state"]["bins"]
            if (set(status) != set(fields.values()) or type(status[fields["hold"]]) is not bool
                    or any(type(status[fields[name]]) is not int for name in ("amount", "limit"))
                    or any(type(value) is not int for value in bins.values())
                    or not set(bins) <= {f"{row['domain']}_bin_{letter}" for letter in "abcdefg"}
                    or not 3 <= len(bins) <= 7 or len(row["priority"]) != len(bins)
                    or set(row["priority"]) != set(bins) or row["state"]["workshop"] != row["domain"]):
                raise ValueError("Invalid visible facts or tie priority")
            targets = independent_gold(row)
            operations = [q["operation"] for q in row["questions"]]
            if len(operations) != len(set(operations)) or not set(operations) <= {"policy", "largest"}:
                raise ValueError("Unexpected question membership")
            # A later leakage filter may remove questions, but must never alter retained ones.
            for q in row["questions"]:
                keys = [key for key, _ in q["options"]]
                expected = set(actions) if q["operation"] == "policy" else set(bins)
                descriptions = (dict(zip(actions, (f"Retain the {row['domain']} items", f"Stage the {row['domain']} items",
                                                    f"Release the {row['domain']} items"))) if q["operation"] == "policy"
                                else {key: f"Select bin {key}" for key in bins})
                if (q["qtype"] != "choice" or q["instructions"] != instructions[q["operation"]]
                        or len(keys) != len(set(keys)) or set(keys) != expected or dict(q["options"]) != descriptions
                        or q["gold_key"] != targets[q["operation"]]):
                    raise ValueError("Choice definition or independent gold mismatch")
                vocabulary[split].update(keys)
    splits = list(vocabulary)
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            if vocabulary[left] & vocabulary[right]:
                raise ValueError("Choice vocabulary overlaps between splits")


def prepare():
    if OUTPUT.exists() or RESULTS.exists():
        raise ValueError("Preserve existing candidate generation")
    data = generate()
    validate(data)
    OUTPUT.mkdir(parents=True)
    RESULTS.mkdir(parents=True)
    for split, rows in data.items():
        write_json(OUTPUT / f"{split}.json", rows)
    manifest = {"status": "unfiltered_do_not_train", "version": "structured-choice-v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
                "generator_sha256": digest(__file__),
                "provenance": "Deterministic rule-derived hard labels; no external examples or model-generated answers are read by the generator. They are not teacher probabilities.",
                "holdout": "Distinct domains, field names, action labels and bin labels across splits. Policy priority and largest-with-ties logic families, generic JSON layout and wording patterns recur. This does not establish unseen-family generalization.",
                "training_permission": "Require the unchanged exact/normalized hash, MinHash, choice-label, embedding and domain-centroid filters against public evaluation, prior authored held-outs, prior probes and the new validation/confirmation sets. No threshold relaxation.",
                "scope": "Candidate preparation only. No training or checkpoint selection. Keep confirmation closed to model evaluation until a subsequent training protocol and checkpoint selection complete.",
                "splits": {split: {"states": len(rows), "decisions": sum(len(r["questions"]) for r in rows),
                                    "sha256": digest(OUTPUT / f"{split}.json"), "domains": list(DOMAINS[split]),
                                    "option_count_distribution": dict(Counter(len(q["options"]) for r in rows for q in r["questions"])),
                                    "gold_option_positions": dict(Counter(next(i for i, (k, _) in enumerate(q["options"]) if k == q["gold_key"])
                                                                         for r in rows for q in r["questions"]))}
                           for split, rows in data.items()}}
    write_json(OUTPUT / "manifest.json", manifest)
    write_json(RESULTS / "generation.json", manifest)
    (OUTPUT / "generation_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    prepare()
