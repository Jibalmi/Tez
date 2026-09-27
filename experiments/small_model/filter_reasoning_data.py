"""Apply existing leakage thresholds to authored training candidates, including the new held-outs.

Run under the shared compute lock: MinHash indexing, MiniLM inference and centroid fitting are heavy.
Any failed check prevents a training-ready export. Choice-option overlap is applied to the fresh
validation/confirmation schemas too; sharing their labels is not silently exempted.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
import re
import time

from reasoning_data import OUTPUT, ROOT, validate
from structure_probe import digest, write_json


def main():
    if "RUN_HEAVY_THREADS" not in os.environ:
        raise RuntimeError("Use C:/temp/run_heavy.py")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import numpy as np
    from common import MINHASH_SEED, minhash, norm_text, state_hashes, thermal_guard
    from datasketch import MinHash, MinHashLSH
    from eval_hashes import embed_texts, load_hashes
    from schemas import eval_option_sets
    from sklearn.cluster import KMeans

    cache = ROOT / "runs/pilot"
    receipt_path = OUTPUT / "filter_receipt.json"
    if receipt_path.exists():
        raise ValueError("Do not overwrite an existing filter attempt")
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    data = {split: json.loads((OUTPUT / f"{split}.json").read_text(encoding="utf-8")) for split in manifest["splits"]}
    for split in data:
        if digest(OUTPUT / f"{split}.json") != manifest["splits"][split]["sha256"]:
            raise ValueError("Authored split hash changed")
    validate(data)
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "stages": [],
               "thresholds": {"minhash": .8, "cosine": .95, "choice_option_overlap": .5},
               "source_sha256": digest(__file__), "source_manifest_sha256": digest(OUTPUT / "manifest.json"),
               "source_files": {}, "drops": []}
    write_json(receipt_path, receipt)
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started > 12 * 60:
            raise TimeoutError("Filter exceeded its 12-minute compute bound")
    try:
        catalog_path = ROOT / "data/small_model/eval_hashes.json.gz"
        catalog_manifest = json.loads((ROOT / "data/small_model/eval_hashes_manifest.json").read_text(encoding="utf-8"))
        if digest(catalog_path) != catalog_manifest["sha256"] or digest(cache / "eval_minhash.npz") != catalog_manifest["minhash"]["sha256"]:
            raise ValueError("Public evaluation guard differs from frozen manifest")
        catalog = load_hashes()
        exact, normalized = set(), set()
        for group in catalog["sources"].values():
            exact.update(group["exact"])
            normalized.update(group["norm"])
        extra = [(f"reasoning_{split}", r["state"]) for split in ("validation", "confirmation") for r in data[split]]
        probe = ROOT / "results/pilot-structure-probe-20260927/cases.json"
        probe_rows = json.loads(probe.read_text(encoding="utf-8"))
        extra += [("structure_probe", r["states"]["nested"]) for r in probe_rows]
        for _, state in extra:
            ex, nm = state_hashes(state)
            exact.add(ex)
            normalized.add(nm)
        rows, seen = [], set()
        for row in data["train"]:
            ex, nm = state_hashes(row["state"])
            reason = "exact_or_normalized_eval_match" if ex in exact or nm in normalized else "duplicate_normalized_train" if nm in seen else None
            if reason:
                receipt["drops"].append({"id": row["id"], "stage": reason})
            else:
                rows.append(row)
                seen.add(nm)
        receipt["stages"].append({"stage": "hash", "states_remaining": len(rows)})
        write_json(receipt_path, receipt)
        print(json.dumps(receipt["stages"][-1]), flush=True)

        cached_mh = np.load(cache / "eval_minhash.npz")
        signatures = np.concatenate([cached_mh["sigs"], np.stack([minhash(norm_text(st)).hashvalues for _, st in extra]).astype(np.uint32)])
        lsh = MinHashLSH(threshold=.8, num_perm=64)
        for i, signature in enumerate(signatures):
            if i % 2048 == 0:
                deadline()
            lsh.insert(str(i), MinHash(num_perm=64, seed=MINHASH_SEED, hashvalues=signature, scheme="affine32"))
        kept = []
        max_jaccard = 0.0
        for row in rows:
            deadline()
            value = minhash(norm_text(row["state"]))
            hits = lsh.query(value)
            similarity = max((float(np.mean(signatures[int(hit)] == value.hashvalues)) for hit in hits), default=0.0)
            max_jaccard = max(max_jaccard, similarity)
            if similarity >= .8:
                receipt["drops"].append({"id": row["id"], "stage": "minhash", "similarity": similarity})
            else:
                kept.append(row)
        rows = kept
        del lsh, signatures
        receipt["stages"].append({"stage": "minhash", "states_remaining": len(rows), "maximum_lsh_candidate_jaccard": max_jaccard})
        write_json(receipt_path, receipt)
        print(json.dumps(receipt["stages"][-1]), flush=True)

        normalize = lambda s: re.sub(r"[\s_\-:.]+", " ", str(s).casefold()).strip()  # noqa: E731
        options = eval_option_sets()
        deadline()
        options += [{normalize(k) for k, _ in q["options"]} for split in ("validation", "confirmation")
                    for r in data[split] for q in r["questions"] if q["qtype"] == "choice"]
        options += [{normalize(k) for k, _ in q["options"]} for r in probe_rows for q in r["questions"] if q["qtype"] == "choice"]
        for row in rows:
            kept_q = []
            for q in row["questions"]:
                labels = {normalize(k) for k, _ in q["options"]}
                overlap = max((len(labels & ev) / len(labels) for ev in options), default=0) if q["qtype"] == "choice" else 0
                if overlap >= .5:
                    receipt["drops"].append({"id": row["id"], "operation": q["operation"], "stage": "schema_overlap", "overlap": overlap})
                else:
                    kept_q.append(q)
            row["questions"] = kept_q
        rows = [row for row in rows if row["questions"]]
        receipt["stages"].append({"stage": "schema", "states_remaining": len(rows), "decisions_remaining": sum(len(r["questions"]) for r in rows)})
        write_json(receipt_path, receipt)
        print(json.dumps(receipt["stages"][-1]), flush=True)
        if not rows:
            raise ValueError("No training candidates survived")

        public_texts = [json.loads(line) for line in (cache / "eval_texts.jsonl").read_text(encoding="utf-8").splitlines()]
        cached_embeddings = np.load(cache / "eval_emb.npz")
        reference = cached_embeddings["emb"].astype(np.float32)
        if len(reference) != len(public_texts) or list(cached_embeddings["src"]) != [r["src"] for r in public_texts]:
            raise ValueError("Evaluation embeddings and source texts are misaligned")
        thermal_guard(force=True)
        deadline()
        anchor = embed_texts([r["text"] for r in public_texts[:32]], "cuda", batch=32).astype(np.float32)
        similarity = (anchor * reference[:32]).sum(1) / (np.linalg.norm(anchor, axis=1) * np.linalg.norm(reference[:32], axis=1))
        receipt["embedding_anchor_min_cosine"] = float(similarity.min())
        if float(similarity.min()) < .999:
            raise ValueError("Current embedder does not reproduce the cached guard anchors")
        thermal_guard(force=True)
        embedded = embed_texts([norm_text(r["state"]) for r in rows] + [norm_text(st) for _, st in extra], "cuda", batch=128).astype(np.float32)
        training_vectors, extra_vectors = embedded[:len(rows)], embedded[len(rows):]
        all_eval = np.concatenate([reference, extra_vectors])
        best = np.full(len(rows), -1., dtype=np.float32)
        for offset in range(0, len(all_eval), 10000):
            deadline()
            best = np.maximum(best, (training_vectors @ all_eval[offset:offset + 10000].T).max(1))
        keep = best < .95
        for row, value in zip(rows, best):
            if value >= .95:
                receipt["drops"].append({"id": row["id"], "stage": "cosine", "similarity": float(value)})
        rows = [r for r, keep_row in zip(rows, keep) if keep_row]
        training_vectors = training_vectors[keep]
        receipt["stages"].append({"stage": "cosine", "states_remaining": len(rows), "maximum_similarity": float(best.max())})
        write_json(receipt_path, receipt)
        print(json.dumps(receipt["stages"][-1]), flush=True)
        if len(rows) < 64:
            raise ValueError("Insufficient states for the registered 64-cluster domain filter")

        sources = list(cached_embeddings["src"]) + [source for source, _ in extra]
        families = np.array([s.split(":")[0] + ":" + s.split(":")[-1] if s.startswith(("typed_decisions", "laya_apps", "h2h"))
                             else s.split(":")[0] for s in sources])
        centroids = []
        for family in sorted(set(families)):
            center = all_eval[families == family].mean(0)
            centroids.append(center / np.linalg.norm(center))
        fitted = KMeans(n_clusters=64, random_state=13, n_init=4).fit(training_vectors)
        deadline()
        train_centers = fitted.cluster_centers_ / np.linalg.norm(fitted.cluster_centers_, axis=1, keepdims=True)
        near_eval = (training_vectors @ np.stack(centroids).T).max(1) > (training_vectors @ train_centers.T).max(1)
        for row, rejected in zip(rows, near_eval):
            if rejected:
                receipt["drops"].append({"id": row["id"], "stage": "domain_centroid"})
        rows = [r for r, rejected in zip(rows, near_eval) if not rejected]
        receipt["stages"].append({"stage": "domain_centroid", "states_remaining": len(rows)})
        if len(rows) < 500:
            raise ValueError("Fewer than 500 states survived; do not launch the proposed training trial")
        validate({"train": rows, "validation": data["validation"], "confirmation": data["confirmation"]})
        write_json(OUTPUT / "train.filtered.json", rows)
        for path in (catalog_path, cache / "eval_minhash.npz", cache / "eval_emb.npz", cache / "eval_texts.jsonl", probe):
            receipt["source_files"][str(path.relative_to(ROOT))] = digest(path)
        receipt.update(status="complete", filtered_train_sha256=digest(OUTPUT / "train.filtered.json"),
                       retained_states=len(rows), retained_decisions=sum(len(r["questions"]) for r in rows),
                       drop_counts=dict(Counter(drop["stage"] for drop in receipt["drops"])))
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        write_json(receipt_path, receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "retained_states", "retained_decisions", "drop_counts")}), flush=True)


if __name__ == "__main__":
    main()
