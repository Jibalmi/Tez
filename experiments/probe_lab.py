"""Probe lab: offline experiments on the cached hidden states of a frozen model (no GPU needed).

The cache (results/probe_cache_<tag>.npz, written by hidden_probe_sweep.py) holds the last-token hidden
state at EVERY layer and the letter logits for the typed-decisions train (6,000) and test (2,000)
decisions. Every experiment below reuses it.

  w2s        label-free probes: train the probe on zero-shot answers (the model's own, or a bigger
             model's) instead of human labels: hard / confidence-filtered / soft labels,
             cross-fitted self-training, cluster-then-label, two-teacher agreement.
  anytime    adaptive depth: read per-layer probes bottom-up and stop when confident
             (max-probability gate, and an SPRT-style accumulated log-odds gate).
  variants   probe families (logreg, shrinkage LDA, kNN, centroid, MLP, multi-layer concat, depth
             pooling) and the size of a decision code (PCA / random projection to m dimensions).
  universal  ONE question-agnostic letter-position probe; leave-one-workflow-out generalisation.
  online     cold start: zero-shot readout -> escalate uncertain decisions -> learn a probe from the
             escalations; accuracy vs human labels spent, against passive random labelling.
  lens       logit lens and DoLa-style layer contrast for the letter readout (final norm + tied head).

  py experiments/probe_lab.py w2s --out results/probe_lab_w2s_qwen35-4b.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("hp", ROOT / "experiments" / "hidden_probe.py")
hp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(hp)  # type: ignore[union-attr]
TEACHER12 = ROOT / "results" / "teacher_labels_gemma4-12b-q8_0"


def softmax(z, axis=-1):
    z = np.asarray(z, float); z = z - z.max(axis=axis, keepdims=True); e = np.exp(z); return e / e.sum(axis=axis, keepdims=True)


def qname(q):
    return f"{q[0]}/{q[1]}"


class Data:
    def __init__(self, tag="qwen35-4b"):
        t0 = time.perf_counter()
        self.train = hp.load_split("train"); self.test = hp.load_split("test")
        d = np.load(ROOT / "results" / f"probe_cache_{tag}.npz")
        self.Htr, self.Hte, self.Ztr, self.Zte = d["H_train"], d["H_test"], d["Z_train"], d["Z_test"]
        self.gtr = np.array([c["gold"] for c in self.train]); self.gte = np.array([c["gold"] for c in self.test])
        self.ktr = np.array([len(c["options"]) for c in self.train]); self.kte = np.array([len(c["options"]) for c in self.test])
        self.qkeys = sorted(set(c["qkey"] for c in self.train))
        self.idx_tr = {q: np.array([i for i, c in enumerate(self.train) if c["qkey"] == q]) for q in self.qkeys}
        self.idx_te = {q: np.array([i for i, c in enumerate(self.test) if c["qkey"] == q]) for q in self.qkeys}
        self.kq = {q: len(self.train[self.idx_tr[q][0]]["options"]) for q in self.qkeys}
        self.qtr = [c["qkey"] for c in self.train]; self.qte = [c["qkey"] for c in self.test]
        self.nL = self.Htr.shape[1]
        self._X = {}
        print(f"loaded {tag}: train {self.Htr.shape}, test {self.Hte.shape}, {len(self.qkeys)} questions, {time.perf_counter() - t0:.0f}s", flush=True)

    def X(self, split, L):
        key = (split, L)
        if key not in self._X:
            self._X[key] = (self.Htr if split == "train" else self.Hte)[:, L].astype(np.float32)
        return self._X[key]

    def teacher(self, name, split):
        """Zero-shot probabilities over the k options, padded to 26."""
        k = self.ktr if split == "train" else self.kte
        if name == "self":
            Z = self.Ztr if split == "train" else self.Zte
            P = np.zeros((len(Z), 26))
            for i in range(len(Z)):
                P[i, :k[i]] = softmax(Z[i, :k[i]])
            return P
        if name == "gemma12b":
            P = np.load(f"{TEACHER12}.{split}.npy").astype(float)
            for i in range(len(P)):
                s = P[i, :k[i]].sum(); P[i, :k[i]] = P[i, :k[i]] / s if s > 0 else 1.0 / k[i]
            return P
        raise ValueError(name)


def masked_argmax(P, k):
    return np.array([int(np.argmax(P[i, :k[i]])) for i in range(len(P))])


def acc_of(P, g, k):
    return float((masked_argmax(P, k) == g).mean())


def fit_lr(X, y, C=0.5, w=None):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=3000, C=C).fit(X, y, sample_weight=w)


def proba(clf, X, k):
    P = np.full((len(X), k), 1e-6); P[:, clf.classes_.astype(int)] += clf.predict_proba(X)
    return P / P.sum(1, keepdims=True)


def probe_predict(D, L, labels, weights=None, C=0.5, Xtr=None, Xte=None, soft=None):
    """Per-question probes on layer L. labels: int per train row (-1 = unlabelled). soft: [N, 26] target
    distributions (soft-label training by weighted duplication). Returns test probabilities [N_te, 26]."""
    Xtr = D.X("train", L) if Xtr is None else Xtr
    Xte = D.X("test", L) if Xte is None else Xte
    P = np.zeros((len(D.test), 26))
    for q in D.qkeys:
        itr, ite, k = D.idx_tr[q], D.idx_te[q], D.kq[q]
        if soft is not None:
            rows, ys, ws = [], [], []
            for j in range(k):
                w = soft[itr, j]; m = w > 1e-3
                rows.append(itr[m]); ys.append(np.full(m.sum(), j)); ws.append(w[m])
            r = np.concatenate(rows); y = np.concatenate(ys); w = np.concatenate(ws)
            if len(np.unique(y)) < 2:
                P[ite, int(y[0])] = 1.0; continue
            P[ite, :k] = proba(fit_lr(Xtr[r], y, C, w), Xte[ite], k); continue
        m = labels[itr] >= 0; r = itr[m]
        if len(r) == 0:
            P[ite, :k] = 1.0 / k; continue
        y = labels[r]
        if len(np.unique(y)) < 2:
            P[ite, int(y[0])] = 1.0; continue
        P[ite, :k] = proba(fit_lr(Xtr[r], y, C, None if weights is None else weights[r]), Xte[ite], k)
    return P


# --------------------------------------------------------------------------------------------- w2s
def exp_w2s(D, args):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    res = {"gold": {}, "teachers": {}}
    layers = [int(x) for x in args.layers.split(",")]
    for L in layers:
        res["gold"][str(L)] = acc_of(probe_predict(D, L, D.gtr), D.gte, D.kte)
        print(f"gold-label probe L{L}: {res['gold'][str(L)]:.3f}", flush=True)
    teachers = ["self"] + (["gemma12b"] if Path(f"{TEACHER12}.train.npy").exists() else [])
    for tname in teachers + (["agree"] if len(teachers) == 2 else []):
        if tname == "agree":
            A, B = D.teacher("self", "train"), D.teacher("gemma12b", "train")
            Ptr = B.copy(); ya, yb = masked_argmax(A, D.ktr), masked_argmax(B, D.ktr)
            agree = ya == yb
            Pte = D.teacher("gemma12b", "test")
        else:
            Ptr, Pte = D.teacher(tname, "train"), D.teacher(tname, "test")
            agree = None
        yt = masked_argmax(Ptr, D.ktr); conf = Ptr.max(1)
        T = {"teacher_train_acc": float((yt == D.gtr).mean()), "teacher_test_acc": acc_of(Pte, D.gte, D.kte), "variants": {}}
        print(f"[{tname}] teacher train acc {T['teacher_train_acc']:.3f}, test acc {T['teacher_test_acc']:.3f}", flush=True)
        for L in layers:
            V = {}

            def rec(name, P, labels=None):
                a = acc_of(P, D.gte, D.kte)
                la = None if labels is None else float((labels[labels >= 0] == D.gtr[labels >= 0]).mean())
                n = None if labels is None else int((labels >= 0).sum())
                V[name] = dict(acc=a, label_acc=la, n_labels=n)
                print(f"  [{tname}] L{L} {name:28s} acc {a:.3f}  (labels: n={n}, correct={la})", flush=True)

            if agree is not None:
                lab = np.where(agree, yt, -1); rec("agreement-filtered", probe_predict(D, L, lab), lab)
                V_ = V; T["variants"][str(L)] = V_; continue
            rec("hard", probe_predict(D, L, yt), yt)
            for keep in (0.5, 0.25):
                lab = np.full(len(yt), -1)
                for q in D.qkeys:
                    itr = D.idx_tr[q]; thr = np.quantile(conf[itr], 1 - keep); m = itr[conf[itr] >= thr]; lab[m] = yt[m]
                rec(f"confident-top{int(keep * 100)}%", probe_predict(D, L, lab), lab)
            rec("soft", probe_predict(D, L, None, soft=Ptr), yt)
            # cross-fitted self-training: relabel each half with a probe fit on the other half
            lab = yt.copy(); Xtr = D.X("train", L); rng = np.random.default_rng(0)
            for rnd in range(3):
                new = lab.copy()
                for q in D.qkeys:
                    itr = D.idx_tr[q]; k = D.kq[q]; perm = rng.permutation(itr); halves = (perm[: len(perm) // 2], perm[len(perm) // 2:])
                    for a_, b_ in ((0, 1), (1, 0)):
                        fit_rows, pred_rows = halves[a_], halves[b_]
                        y = lab[fit_rows]
                        if len(np.unique(y)) < 2:
                            new[pred_rows] = y[0]; continue
                        Pp = proba(fit_lr(Xtr[fit_rows], y), Xtr[pred_rows], k)
                        new[pred_rows] = np.argmax(Pp, 1)
                lab = new
                rec(f"self-train round {rnd + 1}", probe_predict(D, L, lab), lab)
            # cluster-then-label: k-means per question on PCA-32 features, clusters named by teacher vote
            lab = np.full(len(yt), -1)
            for q in D.qkeys:
                itr = D.idx_tr[q]; k = D.kq[q]
                Zq = PCA(n_components=32, random_state=0).fit_transform(Xtr[itr])
                km = KMeans(n_clusters=min(3 * k, len(itr) // 5), n_init=5, random_state=0).fit(Zq)
                for c in np.unique(km.labels_):
                    m = itr[km.labels_ == c]; lab[m] = int(np.argmax(Ptr[m, :k].sum(0)))
            rec("cluster-then-label", probe_predict(D, L, lab), lab)
            T["variants"][str(L)] = V
        res["teachers"][tname] = T
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ----------------------------------------------------------------------------------------- anytime
def exp_anytime(D, args):
    grid = [int(x) for x in args.grid.split(",")]
    full = D.nL - 1
    Ps = {}
    for L in grid:
        Ps[L] = probe_predict(D, L, D.gtr); print(f"L{L}: {acc_of(Ps[L], D.gte, D.kte):.3f}", flush=True)
    res = {"grid": grid, "full_depth": full, "fixed": {str(L): acc_of(Ps[L], D.gte, D.kte) for L in grid}, "gate": {}, "accumulate": {}}
    n = len(D.test)
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        pred = np.zeros(n, int); depth = np.zeros(n)
        for i in range(n):
            k = D.kte[i]
            for L in grid:
                p = Ps[L][i, :k]
                if p.max() >= tau or L == grid[-1]:
                    pred[i] = int(np.argmax(p)); depth[i] = L; break
        res["gate"][str(tau)] = dict(acc=float((pred == D.gte).mean()), mean_depth=float(depth.mean()), speedup=float(full / depth.mean()))
        print(f"gate tau={tau}: acc {res['gate'][str(tau)]['acc']:.3f} mean depth {depth.mean():.1f} ({full / depth.mean():.2f}x)", flush=True)
    for theta in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
        pred = np.zeros(n, int); depth = np.zeros(n)
        for i in range(n):
            k = D.kte[i]; acc_lp = np.zeros(k)
            for L in grid:
                acc_lp += np.log(np.clip(Ps[L][i, :k], 1e-9, 1)); s = np.sort(acc_lp)[::-1]
                if s[0] - s[1] >= theta or L == grid[-1]:
                    pred[i] = int(np.argmax(acc_lp)); depth[i] = L; break
        res["accumulate"][str(theta)] = dict(acc=float((pred == D.gte).mean()), mean_depth=float(depth.mean()), speedup=float(full / depth.mean()))
        print(f"accumulate theta={theta}: acc {res['accumulate'][str(theta)]['acc']:.3f} mean depth {depth.mean():.1f} ({full / depth.mean():.2f}x)", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ---------------------------------------------------------------------------------------- variants
def exp_variants(D, args):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.decomposition import PCA
    L = args.layer
    res = {"layer": L, "families": {}, "code_size": {"pca": {}, "random": {}}}

    def run_family(name, make, Xtr, Xte):
        t0 = time.perf_counter(); P = np.zeros((len(D.test), 26))
        for q in D.qkeys:
            itr, ite, k = D.idx_tr[q], D.idx_te[q], D.kq[q]; y = D.gtr[itr]
            if len(np.unique(y)) < 2:
                P[ite, int(y[0])] = 1.0; continue
            try:
                clf = make().fit(Xtr[itr], y); P[ite, :k] = proba(clf, Xte[ite], k)
            except Exception as e:  # noqa: BLE001  (e.g. a class with one member under early stopping)
                print(f"   {name} failed on {qname(q)}: {str(e)[:80]}; falling back to logreg", flush=True)
                P[ite, :k] = proba(fit_lr(Xtr[itr], y), Xte[ite], k)
        a = acc_of(P, D.gte, D.kte); res["families"][name] = dict(acc=a, seconds=time.perf_counter() - t0)
        print(f"{name:34s} {a:.3f}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    from sklearn.linear_model import LogisticRegression
    Xtr, Xte = D.X("train", L), D.X("test", L)
    nrm = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)  # noqa: E731
    run_family("logreg C=0.5", lambda: LogisticRegression(max_iter=3000, C=0.5), Xtr, Xte)
    run_family("logreg C=0.05", lambda: LogisticRegression(max_iter=3000, C=0.05), Xtr, Xte)
    run_family("logreg C=5", lambda: LogisticRegression(max_iter=3000, C=5.0), Xtr, Xte)
    run_family("shrinkage LDA", lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), Xtr, Xte)
    run_family("kNN cosine k=15", lambda: KNeighborsClassifier(n_neighbors=15, weights="distance"), nrm(Xtr), nrm(Xte))

    class Centroid:
        def fit(self, X, y):
            self.classes_ = np.unique(y); self.C = np.stack([nrm(X[y == c].mean(0, keepdims=True))[0] for c in self.classes_]); return self

        def predict_proba(self, X):
            return softmax(nrm(X) @ self.C.T * 20.0, 1)
    run_family("nearest centroid (cosine)", Centroid, Xtr, Xte)
    run_family("MLP 256 (early stop)", lambda: MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, early_stopping=True, max_iter=300, random_state=0), Xtr, Xte)
    cat = [18, 22, 26, 30]
    run_family("logreg on concat L18+22+26+30", lambda: LogisticRegression(max_iter=3000, C=0.5),
               np.concatenate([D.X("train", l) for l in cat], 1), np.concatenate([D.X("test", l) for l in cat], 1))
    pool = list(range(20, 29))
    run_family("logreg on mean of L20-28", lambda: LogisticRegression(max_iter=3000, C=0.5),
               np.mean([D.X("train", l) for l in pool], 0), np.mean([D.X("test", l) for l in pool], 0))
    # decision code size
    d = Xtr.shape[1]
    pca = PCA(n_components=512, random_state=0).fit(Xtr)
    Ttr, Tte = pca.transform(Xtr), pca.transform(Xte)
    rng = np.random.default_rng(0); R = rng.standard_normal((d, 512)).astype(np.float32) / math.sqrt(512)
    Rtr, Rte = Xtr @ R, Xte @ R
    for m in (2, 4, 8, 16, 32, 64, 128, 256, 512):
        for kind, A, B in (("pca", Ttr, Tte), ("random", Rtr, Rte)):
            P = np.zeros((len(D.test), 26))
            for q in D.qkeys:
                itr, ite, k = D.idx_tr[q], D.idx_te[q], D.kq[q]; y = D.gtr[itr]
                if len(np.unique(y)) < 2:
                    P[ite, int(y[0])] = 1.0; continue
                P[ite, :k] = proba(LogisticRegression(max_iter=3000, C=0.5).fit(A[itr, :m], y), B[ite, :m], k)
            res["code_size"][kind][str(m)] = acc_of(P, D.gte, D.kte)
        print(f"code size m={m:4d}: PCA {res['code_size']['pca'][str(m)]:.3f}  random {res['code_size']['random'][str(m)]:.3f}", flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    res["code_size"]["full"] = res["families"]["logreg C=0.5"]["acc"]; res["code_size"]["dim"] = int(d)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# --------------------------------------------------------------------------------------- universal
def exp_universal(D, args):
    import torch
    torch.manual_seed(0)
    wfs = sorted(set(q[0] for q in D.qkeys))
    wf_tr = np.array([q[0] for q in D.qtr]); wf_te = np.array([q[0] for q in D.qte])
    T12 = D.teacher("gemma12b", "test") if Path(f"{TEACHER12}.test.npy").exists() else None
    S4 = D.teacher("self", "test")

    def train_eval(L, tr_rows, te_rows, wd, epochs=200):
        X = torch.tensor(D.X("train", L)[tr_rows]); mu, sd = X.mean(0), X.std(0) + 1e-3; X = (X - mu) / sd
        y = torch.tensor(D.gtr[tr_rows]); km = torch.arange(26)[None, :] < torch.tensor(D.ktr[tr_rows])[:, None]
        W = torch.zeros(X.shape[1], 26, requires_grad=True); b = torch.zeros(26, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=5e-3)
        for _ in range(epochs):
            opt.zero_grad(); lg = (X @ W + b).masked_fill(~km, -1e9)
            loss = torch.nn.functional.cross_entropy(lg, y) + wd * (W ** 2).sum(); loss.backward(); opt.step()
        Xt = (torch.tensor(D.X("test", L)[te_rows]) - mu) / sd; kt = torch.arange(26)[None, :] < torch.tensor(D.kte[te_rows])[:, None]
        with torch.no_grad():
            pr = (Xt @ W + b).masked_fill(~kt, -1e9).argmax(1).numpy()
        return float((pr == D.gte[te_rows]).mean())

    res = {"layers": {}, "baselines": {}}
    for w in wfs:
        te_rows = np.where(wf_te == w)[0]
        res["baselines"][w] = dict(letters_4b=float((masked_argmax(S4[te_rows], D.kte[te_rows]) == D.gte[te_rows]).mean()),
                                   letters_12b=None if T12 is None else float((masked_argmax(T12[te_rows], D.kte[te_rows]) == D.gte[te_rows]).mean()))
    for L in [int(x) for x in args.layers.split(",")]:
        R = {"lowo": {}, "in_distribution": None}
        R["in_distribution"] = train_eval(L, np.arange(len(D.train)), np.arange(len(D.test)), 1e-3)
        for w in wfs:
            tr_w = [x for x in wfs if x != w]
            best, best_wd = -1, None
            for wd in (1e-4, 1e-3, 1e-2):   # inner leave-one-workflow-out on the three training workflows
                sc = np.mean([train_eval(L, np.where(np.isin(wf_tr, [x for x in tr_w if x != v]))[0], np.where(wf_te == v)[0], wd, 150) for v in tr_w])
                if sc > best:
                    best, best_wd = sc, wd
            a = train_eval(L, np.where(wf_tr != w)[0], np.where(wf_te == w)[0], best_wd)
            R["lowo"][w] = dict(acc=a, wd=best_wd, inner=best)
            print(f"L{L} held-out {w:28s} universal {a:.3f} (wd {best_wd}) | 4B letters {res['baselines'][w]['letters_4b']:.3f} | 12B letters {res['baselines'][w]['letters_12b']}", flush=True)
        R["lowo_mean"] = float(np.mean([v["acc"] for v in R["lowo"].values()]))
        print(f"L{L}: in-distribution {R['in_distribution']:.3f}, leave-one-workflow-out mean {R['lowo_mean']:.3f}", flush=True)
        res["layers"][str(L)] = R
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ------------------------------------------------------------------------------------------ online
def exp_online(D, args):
    L = args.layer
    Xtr, Xte = D.X("train", L), D.X("test", L)
    res = {"layer": L, "teachers": {}}
    teachers = ["self"] + (["gemma12b"] if Path(f"{TEACHER12}.train.npy").exists() else [])
    for tname in teachers:
        Ptr, Pte = D.teacher(tname, "train"), D.teacher(tname, "test")
        TR = {"zero_shot_test_acc": acc_of(Pte, D.gte, D.kte), "policies": {}, "passive": {}}
        for tau in (0.6, 0.8, 0.9, 0.95):
            for blend0 in (20,):
                rng = np.random.default_rng(args.seed); order = rng.permutation(len(D.train))
                lab = {q: [] for q in D.qkeys}; model = {q: None for q in D.qkeys}; nfit = {q: 0 for q in D.qkeys}
                auto_ok = auto_n = esc = 0; curve = []

                def current_probs(i, q, X):
                    k = D.kq[q]; p0 = (Ptr if X is Xtr else Pte)[i, :k]
                    if model[q] is None:
                        return p0
                    n = len(lab[q]); w = n / (n + blend0)
                    return w * proba(model[q], X[i:i + 1], k)[0] + (1 - w) * p0

                for step, i in enumerate(order):
                    q = D.qtr[i]; p = current_probs(i, q, Xtr)
                    if p.max() < tau:
                        esc += 1; lab[q].append(i)
                        if len(lab[q]) - nfit[q] >= 5 and len(lab[q]) >= 6 and len(np.unique(D.gtr[lab[q]])) >= 2:
                            model[q] = fit_lr(Xtr[lab[q]], D.gtr[lab[q]]); nfit[q] = len(lab[q])
                    else:
                        auto_n += 1; auto_ok += int(np.argmax(p) == D.gtr[i])
                    if (step + 1) % 500 == 0 or step + 1 == len(order):
                        Pt = np.zeros((len(D.test), 26))
                        for qq in D.qkeys:
                            ite = D.idx_te[qq]; k = D.kq[qq]
                            if model[qq] is None:
                                Pt[ite, :k] = Pte[ite, :k]
                            else:
                                n = len(lab[qq]); w = n / (n + blend0)
                                Pt[ite, :k] = w * proba(model[qq], Xte[ite], k) + (1 - w) * Pte[ite, :k]
                        curve.append(dict(step=step + 1, labels=esc, auto_rate=auto_n / (step + 1), auto_acc=auto_ok / max(1, auto_n),
                                          system_acc=(auto_ok + esc) / (step + 1), test_acc=acc_of(Pt, D.gte, D.kte)))
                TR["policies"][f"tau{tau}"] = curve
                c = curve[-1]
                print(f"[{tname}] tau {tau}: labels {c['labels']} ({c['labels'] / len(order):.0%}), auto acc {c['auto_acc']:.3f} on {c['auto_rate']:.0%}, "
                      f"stream acc {c['system_acc']:.3f}, held-out test acc {c['test_acc']:.3f}", flush=True)
                # passive baseline: the same number of labels, drawn at random, probe blended the same way
                budget = c["labels"]; ridx = np.random.default_rng(args.seed + 1).permutation(len(D.train))[:budget]
                labels = np.full(len(D.train), -1); labels[ridx] = D.gtr[ridx]
                Pp = probe_predict(D, L, labels, Xtr=Xtr, Xte=Xte)
                Pb = np.zeros_like(Pp)
                for q in D.qkeys:
                    ite = D.idx_te[q]; n = int((labels[D.idx_tr[q]] >= 0).sum()); w = n / (n + blend0); k = D.kq[q]
                    Pb[ite, :k] = w * Pp[ite, :k] + (1 - w) * Pte[ite, :k]
                TR["passive"][f"tau{tau}"] = dict(labels=budget, test_acc=acc_of(Pb, D.gte, D.kte))
                print(f"    passive random labelling with {budget} labels: held-out test acc {TR['passive'][f'tau{tau}']['test_acc']:.3f}", flush=True)
                res["teachers"][tname] = TR
                Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# -------------------------------------------------------------------------------------------- lens
def exp_lens(D, args):
    import glob
    from safetensors import safe_open
    from transformers import AutoTokenizer
    gemma = args.tag.startswith("gemma")
    repo = "google/gemma-4-12B-it" if gemma else "Qwen/Qwen3.5-4B"
    snap = glob.glob(str(Path.home() / ".cache/huggingface/hub" / ("models--" + repo.replace("/", "--")) / "snapshots/*"))[0]
    tok = AutoTokenizer.from_pretrained(repo)
    ids = [tok.encode(Lt, add_special_tokens=False)[0] for Lt in hp.LETTERS]
    Wn = E = None
    for f in glob.glob(snap + "/*.safetensors"):
        with safe_open(f, "pt") as s_:
            ks = set(s_.keys())
            if "model.language_model.norm.weight" in ks:   # slice reads: whole-tensor reads of a 24 GB file crash on Windows
                Wn = s_.get_slice("model.language_model.norm.weight")[:].float().numpy()
            if "model.language_model.embed_tokens.weight" in ks:
                sl = s_.get_slice("model.language_model.embed_tokens.weight")
                E = np.concatenate([sl[i:i + 1].float().numpy() for i in ids], 0)
    eps = 1e-6
    scale = Wn if gemma else (1.0 + Wn)          # Gemma 4 RMSNorm multiplies by w; Qwen3.5 by (1 + w)
    cap = 30.0 if gemma else None                # Gemma final-logit soft-capping (monotonic)

    def lens(H):
        H = H.astype(np.float32); h = H / np.sqrt((H ** 2).mean(-1, keepdims=True) + eps) * scale
        z = h @ E.T
        return cap * np.tanh(z / cap) if cap else z

    full = D.nL - 1
    Zf_raw = D.Hte[:, full].astype(np.float32) @ E.T; Zf_raw = cap * np.tanh(Zf_raw / cap) if cap else Zf_raw; Zf_norm = lens(D.Hte[:, full])
    corr = lambda A, B: float(np.corrcoef(A[B > -1e3], B[B > -1e3])[0, 1])  # noqa: E731  (the cache pads unused letter slots with -1e4)
    res = {"check": {"corr_raw_vs_cached": corr(Zf_raw, D.Zte), "corr_normed_vs_cached": corr(Zf_norm, D.Zte)}}
    post_norm = res["check"]["corr_raw_vs_cached"] > res["check"]["corr_normed_vs_cached"]
    res["check"]["last_hidden_state_is_post_norm"] = bool(post_norm)
    print("check", res["check"], flush=True)
    Zl = {}
    for L in range(0, D.nL):
        Zl[L] = Zf_raw if (L == full and post_norm) else lens(D.Hte[:, L])
    res["logit_lens"] = {str(L): acc_of(np.stack([np.pad(softmax(Zl[L][i, :D.kte[i]]), (0, 26 - D.kte[i])) for i in range(len(D.test))]), D.gte, D.kte) for L in Zl}
    print("logit lens acc by layer:", {k: round(v, 3) for k, v in res["logit_lens"].items()}, flush=True)
    res["dola"] = {}
    for M in range(4, full, 2):
        sc = []
        for i in range(len(D.test)):
            k = D.kte[i]; a = Zl[full][i, :k] - np.logaddexp.reduce(Zl[full][i, :k]); b = Zl[M][i, :k] - np.logaddexp.reduce(Zl[M][i, :k])
            sc.append(np.pad(softmax(a - b), (0, 26 - k)))
        res["dola"][str(M)] = acc_of(np.stack(sc), D.gte, D.kte)
    print("DoLa (final minus layer M) acc:", {k: round(v, 3) for k, v in res["dola"].items()}, flush=True)
    # lens-averaged readout over the plateau layers
    for lo, hi in ((20, 28), (24, 32), (16, 32)):
        sc = np.stack([np.pad(softmax(np.mean([Zl[L][i, :D.kte[i]] - np.logaddexp.reduce(Zl[L][i, :D.kte[i]]) for L in range(lo, hi + 1)], 0)), (0, 26 - D.kte[i])) for i in range(len(D.test))])
        res[f"lens_mean_{lo}_{hi}"] = acc_of(sc, D.gte, D.kte)
        print(f"lens mean L{lo}-{hi}: {res[f'lens_mean_{lo}_{hi}']:.3f}", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ------------------------------------------------------------------------------------ label model
def lens_probs(D, L, split):
    """Zero-shot letter probabilities read at layer L through the final norm and tied head (Qwen3.5 only)."""
    import glob
    from safetensors import safe_open
    from transformers import AutoTokenizer
    if not hasattr(D, "_lens"):
        snap = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*"))[0]
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B"); ids = [tok.encode(Lt, add_special_tokens=False)[0] for Lt in hp.LETTERS]
        Wn = E = None
        for f in glob.glob(snap + "/*.safetensors"):
            with safe_open(f, "pt") as s_:
                ks = set(s_.keys())
                if "model.language_model.norm.weight" in ks:
                    Wn = s_.get_slice("model.language_model.norm.weight")[:].float().numpy()
                if "model.language_model.embed_tokens.weight" in ks:
                    sl = s_.get_slice("model.language_model.embed_tokens.weight"); E = np.concatenate([sl[i:i + 1].float().numpy() for i in ids], 0)
        D._lens = (Wn, E)
    Wn, E = D._lens
    H = (D.Htr if split == "train" else D.Hte)[:, L].astype(np.float32); k = D.ktr if split == "train" else D.kte
    Z = (H / np.sqrt((H ** 2).mean(-1, keepdims=True) + 1e-6) * (1.0 + Wn)) @ E.T
    P = np.zeros((len(H), 26))
    for i in range(len(H)):
        P[i, :k[i]] = softmax(Z[i, :k[i]])
    return P


def dawid_skene(votes, k, iters=100, init=None):
    """votes: [n, m] hard labels from m readouts. Returns (posteriors [n, k], class prior, confusions [m, k, k])."""
    n, m = votes.shape
    if init is None:
        T = np.zeros((n, k))
        for j in range(m):
            T[np.arange(n), votes[:, j]] += 1
        T /= T.sum(1, keepdims=True)
    else:
        T = init.copy()
    for _ in range(iters):
        pi = (T.sum(0) + 1) / (n + k)
        th = np.zeros((m, k, k))
        for j in range(m):
            for l in range(k):
                th[j, :, l] = T[votes[:, j] == l].sum(0)
            th[j] = (th[j] + 1) / (th[j].sum(1, keepdims=True) + k)
        logT = np.log(pi)[None, :] + sum(np.log(th[j][:, votes[:, j]]).T for j in range(m))
        T = softmax(logT, 1)
    return T, pi, th


def ds_apply(votes, pi, th):
    logT = np.log(pi)[None, :] + sum(np.log(th[j][:, votes[:, j]]).T for j in range(votes.shape[1]))
    return softmax(logT, 1)


def exp_labelmodel(D, args):
    views = {"12B letters": (D.teacher("gemma12b", "train"), D.teacher("gemma12b", "test")),
             "4B letters": (D.teacher("self", "train"), D.teacher("self", "test")),
             "4B lens L29": (lens_probs(D, 29, "train"), lens_probs(D, 29, "test")),
             "4B lens L24": (lens_probs(D, 24, "train"), lens_probs(D, 24, "test"))}
    res = {"views_test_acc": {v: acc_of(P[1], D.gte, D.kte) for v, P in views.items()}, "combos": {}}
    print("views:", {k: round(v, 3) for k, v in res["views_test_acc"].items()}, flush=True)
    combos = {"12B + 4B letters + lens29": ["12B letters", "4B letters", "4B lens L29"],
              "12B + lens29 + lens24": ["12B letters", "4B lens L29", "4B lens L24"],
              "all four": list(views)}
    for cname, vs in combos.items():
        Ttr = np.zeros((len(D.train), 26)); Tte = np.zeros((len(D.test), 26)); MVte = np.zeros((len(D.test), 26))
        for q in D.qkeys:
            itr, ite, k = D.idx_tr[q], D.idx_te[q], D.kq[q]
            vtr = np.stack([masked_argmax(views[v][0][itr], D.ktr[itr]) for v in vs], 1)
            vte = np.stack([masked_argmax(views[v][1][ite], D.kte[ite]) for v in vs], 1)
            init = views[vs[0]][0][itr, :k] * 0.5 + 0.5 / k    # break symmetry toward the strongest readout
            T, pi, th = dawid_skene(vtr, k, init=init)
            Ttr[itr, :k] = T; Tte[ite, :k] = ds_apply(vte, pi, th)
            for j in range(len(vs)):
                MVte[ite, vte[:, j]] += 1
        r = {"label_model_test_acc": acc_of(Tte, D.gte, D.kte), "majority_vote_test_acc": acc_of(MVte + 1e-3 * views[vs[0]][1], D.gte, D.kte),
             "label_model_train_label_acc": float((masked_argmax(Ttr, D.ktr) == D.gtr).mean())}
        for L in (22, 26):
            yl = masked_argmax(Ttr, D.ktr)
            r[f"probe_L{L}_hard"] = acc_of(probe_predict(D, L, yl), D.gte, D.kte)
            r[f"probe_L{L}_soft"] = acc_of(probe_predict(D, L, None, soft=Ttr), D.gte, D.kte)
        res["combos"][cname] = r
        print(cname, {k: round(v, 3) for k, v in r.items()}, flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# --------------------------------------------------------------------------------------- few-shot
def exp_fewshot(D, args):
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    L = args.layer; Xtr, Xte = D.X("train", L), D.X("test", L)
    stats = {}   # unlabelled statistics per question (no labels used): shrunk covariance and a PCA-64 basis
    for q in D.qkeys:
        Xq = Xtr[D.idx_tr[q]]
        lw = LedoitWolf().fit(Xq); Pm = np.linalg.inv(lw.covariance_)
        stats[q] = (Pm, PCA(n_components=64, random_state=0).fit(Xq))
    res = {"layer": L, "n": {}}
    methods = ("logreg", "LDA (labelled covariance)", "LDA (unlabelled covariance)", "PCA-64 (unlabelled) + logreg")
    for n in (5, 10, 25, 50):
        accs = {m: [] for m in methods}
        for seed in range(3):
            rng = np.random.default_rng(seed); ok = {m: 0 for m in methods}
            for q in D.qkeys:
                itr, ite = D.idx_tr[q], D.idx_te[q]
                pick = rng.choice(itr, min(n, len(itr)), replace=False); y = D.gtr[pick]; yt = D.gte[ite]
                if len(np.unique(y)) < 2:
                    for m in ok:
                        ok[m] += int((yt == y[0]).sum())
                    continue
                ok["logreg"] += int((LogisticRegression(max_iter=3000, C=0.5).fit(Xtr[pick], y).predict(Xte[ite]) == yt).sum())
                try:
                    ok["LDA (labelled covariance)"] += int((LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr[pick], y).predict(Xte[ite]) == yt).sum())
                except Exception:  # noqa: BLE001
                    ok["LDA (labelled covariance)"] += int((LogisticRegression(max_iter=3000, C=0.5).fit(Xtr[pick], y).predict(Xte[ite]) == yt).sum())
                Pm, pca = stats[q]; cls = np.unique(y)
                M = np.stack([Xtr[pick][y == c].mean(0) for c in cls]); W = M @ Pm
                b = -0.5 * np.sum(W * M, 1) + np.log(np.array([(y == c).mean() for c in cls]))
                ok["LDA (unlabelled covariance)"] += int((cls[np.argmax(Xte[ite] @ W.T + b, 1)] == yt).sum())
                ok["PCA-64 (unlabelled) + logreg"] += int((LogisticRegression(max_iter=3000, C=0.5).fit(pca.transform(Xtr[pick]), y).predict(pca.transform(Xte[ite])) == yt).sum())
            for m in ok:
                accs[m].append(ok[m] / len(D.test))
        res["n"][str(n)] = {m: dict(mean=float(np.mean(v)), std=float(np.std(v))) for m, v in accs.items()}
        print(f"n={n:3d}/question: " + " | ".join(f"{m} {np.mean(v):.3f}" for m, v in accs.items()), flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ------------------------------------------------------------------------- conformal selection (FDR)
def exp_confsel(D, args):
    readouts = {"4B probe L26 (gold labels)": probe_predict(D, 26, D.gtr)}
    if Path(f"{TEACHER12}.test.npy").exists():
        readouts["12B letters (zero-shot)"] = D.teacher("gemma12b", "test")
        readouts["4B probe L26 (12B pseudo-labels, no human labels)"] = probe_predict(D, 26, masked_argmax(D.teacher("gemma12b", "train"), D.ktr))
    res = {}
    for name, P in readouts.items():
        pred = masked_argmax(P, D.kte); conf = np.array([P[i, :D.kte[i]].max() for i in range(len(P))]); wrong = (pred != D.gte)
        R = {"accuracy": float(1 - wrong.mean())}
        for alpha in (0.05, 0.1, 0.2):
            acted, fdr = [], []
            for rep in range(50):
                rng = np.random.default_rng(rep); perm = rng.permutation(len(P)); cal, ev = perm[: len(P) // 2], perm[len(P) // 2:]
                sw = np.sort(conf[cal][wrong[cal]])
                pv = (1 + (len(sw) - np.searchsorted(sw, conf[ev], side="left"))) / (len(cal) + 1)   # p-value of H0 "this decision is wrong"
                order = np.argsort(pv); m = len(ev); thr = alpha * np.arange(1, m + 1) / m
                okk = np.where(pv[order] <= thr)[0]; sel = order[: okk.max() + 1] if len(okk) else np.array([], int)
                acted.append(len(sel) / m); fdr.append(float(wrong[ev][sel].mean()) if len(sel) else 0.0)
            R[f"alpha{alpha}"] = dict(acted=float(np.mean(acted)), realised_error_among_acted=float(np.mean(fdr)), worst_error=float(np.max(fdr)))
        res[name] = R
        print(name, json.dumps(R), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ---------------------------------------------------------------------------------- stacked pooling
def crossfit_probe_train(D, L, labels, folds=5, seed=0):
    """Out-of-fold probe probabilities on the TRAIN split (for stacking without leakage)."""
    Xtr = D.X("train", L); P = np.zeros((len(D.train), 26)); rng = np.random.default_rng(seed)
    for q in D.qkeys:
        itr, k = D.idx_tr[q], D.kq[q]; perm = rng.permutation(itr); parts = np.array_split(perm, folds)
        for f in range(folds):
            te_ = parts[f]; tr_ = np.concatenate([parts[g] for g in range(folds) if g != f]); y = labels[tr_]
            if len(np.unique(y)) < 2:
                P[te_, int(y[0])] = 1.0; continue
            P[te_, :k] = proba(fit_lr(Xtr[tr_], y), Xtr[te_], k)
    return P


def exp_stack(D, args):
    """Out-of-fold log-linear pooling of every readout (probe, lens, 4B letters, 12B letters), per question."""
    from sklearn.linear_model import LogisticRegression
    src = {"probe L26": (crossfit_probe_train(D, 26, D.gtr), probe_predict(D, 26, D.gtr)),
           "probe L22": (crossfit_probe_train(D, 22, D.gtr), probe_predict(D, 22, D.gtr)),
           "lens L29": (lens_probs(D, 29, "train"), lens_probs(D, 29, "test")),
           "4B letters": (D.teacher("self", "train"), D.teacher("self", "test"))}
    if Path(f"{TEACHER12}.train.npy").exists():
        src["12B letters"] = (D.teacher("gemma12b", "train"), D.teacher("gemma12b", "test"))
    res = {"single": {n: acc_of(v[1], D.gte, D.kte) for n, v in src.items()}, "pools": {}}
    print("single sources:", {k: round(v, 3) for k, v in res["single"].items()}, flush=True)
    pools = {"probe L26 + 12B letters": ["probe L26", "12B letters"], "probe L26 + 4B letters": ["probe L26", "4B letters"],
             "probe L26 + L22 + lens + 4B + 12B": list(src)}
    for pname, names in pools.items():
        if any(n not in src for n in names):
            continue
        P = np.zeros((len(D.test), 26)); Pg = np.zeros((len(D.test), 26))
        for q in D.qkeys:
            itr, ite, k = D.idx_tr[q], D.idx_te[q], D.kq[q]
            ftr = np.concatenate([np.log(np.clip(src[n][0][itr, :k], 1e-6, 1)) for n in names], 1)
            fte = np.concatenate([np.log(np.clip(src[n][1][ite, :k], 1e-6, 1)) for n in names], 1)
            y = D.gtr[itr]
            if len(np.unique(y)) < 2:
                P[ite, int(y[0])] = 1.0; Pg[ite, int(y[0])] = 1.0; continue
            P[ite, :k] = proba(LogisticRegression(max_iter=3000, C=1.0).fit(ftr, y), fte, k)
            # geometric pool with one exponent per source (extremised log-pooling), fitted by grid on train
            best = None
            for w in np.linspace(0, 2, 9):
                for w2 in np.linspace(0, 2, 9):
                    ws = [1.0] + [w] + [w2] * (len(names) - 2) if len(names) > 2 else [1.0, w]
                    s_ = sum(wj * np.log(np.clip(src[n][0][itr, :k], 1e-6, 1)) for wj, n in zip(ws, names))
                    a = float((np.argmax(s_, 1) == y).mean())
                    if best is None or a > best[0]:
                        best = (a, ws)
                    if len(names) == 2:
                        break
            st = sum(wj * np.log(np.clip(src[n][1][ite, :k], 1e-6, 1)) for wj, n in zip(best[1], names))
            Pg[ite, :k] = softmax(st, 1)
        res["pools"][pname] = dict(stacked_logreg=acc_of(P, D.gte, D.kte), geometric_pool=acc_of(Pg, D.gte, D.kte))
        print(pname, res["pools"][pname], flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# --------------------------------------------------------------------- typicality-first cold start
def exp_typical(D, args):
    """Which rows should a human label first? Random vs typicality (TypiClust-style k-means centres on unlabelled states)."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    L = args.layer; Xtr, Xte = D.X("train", L), D.X("test", L)
    res = {"layer": L, "n": {}}
    for n in (5, 10, 25):
        out = {"random": [], "typical": []}
        for seed in range(3):
            rng = np.random.default_rng(seed); ok = {"random": 0, "typical": 0}
            for q in D.qkeys:
                itr, ite = D.idx_tr[q], D.idx_te[q]; yt = D.gte[ite]
                Z = PCA(n_components=32, random_state=seed).fit_transform(Xtr[itr])
                km = KMeans(n_clusters=n, n_init=3, random_state=seed).fit(Z)
                typ = []
                for c in range(n):   # most typical member of each cluster: highest inverse mean distance to its 20 nearest in-cluster neighbours
                    m = np.where(km.labels_ == c)[0]
                    if len(m) == 0:
                        continue
                    d = np.linalg.norm(Z[m][:, None] - Z[m][None], axis=-1); kk = min(20, len(m))
                    t = 1.0 / (np.sort(d, 1)[:, 1:kk].mean(1) + 1e-6) if len(m) > 1 else np.ones(1)
                    typ.append(itr[m[int(np.argmax(t))]])
                picks = {"random": rng.choice(itr, n, replace=False), "typical": np.array(typ)}
                for name, pick in picks.items():
                    y = D.gtr[pick]
                    if len(np.unique(y)) < 2:
                        ok[name] += int((yt == y[0]).sum()); continue
                    ok[name] += int((fit_lr(Xtr[pick], y).predict(Xte[ite]) == yt).sum())
            for name in out:
                out[name].append(ok[name] / len(D.test))
        res["n"][str(n)] = {k: dict(mean=float(np.mean(v)), std=float(np.std(v))) for k, v in out.items()}
        print(f"n={n}: random {np.mean(out['random']):.3f} | typical {np.mean(out['typical']):.3f}", flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ------------------------------------------------------------------ intrinsic dimension by layer
def exp_intrinsic(D, args):
    """TwoNN intrinsic dimension of the unlabelled states at every layer: does it locate the probe plateau without labels?"""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(0); sub = rng.choice(len(D.train), 2000, replace=False)
    res = {"layers": {}}
    for L in range(0, D.nL):
        X = D.Htr[sub, L].astype(np.float32); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
        dist, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)
        r1, r2 = dist[:, 1], dist[:, 2]; m = (r1 > 1e-9) & (r2 > r1)
        mu = r2[m] / r1[m]; idim = float(m.sum() / np.log(mu).sum())
        # per-question average too (the probes are per question)
        per_q = []
        for q in D.qkeys:
            Xq = D.Htr[D.idx_tr[q], L].astype(np.float32); Xq = Xq / (np.linalg.norm(Xq, axis=1, keepdims=True) + 1e-6)
            dq, _ = NearestNeighbors(n_neighbors=3).fit(Xq).kneighbors(Xq); a, b = dq[:, 1], dq[:, 2]; mm = (a > 1e-9) & (b > a)
            per_q.append(float(mm.sum() / np.log(b[mm] / a[mm]).sum()))
        res["layers"][str(L)] = dict(twonn_pooled=idim, twonn_per_question=float(np.mean(per_q)))
        print(f"L{L:2d}: ID pooled {idim:6.1f} per-question {np.mean(per_q):6.1f}", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ------------------------------------------------------- cold start with a random audit slice
def exp_online2(D, args):
    """Cold start, second design. Escalated labels are a biased sample (only uncertain rows), so a probe
    trained on them can lose to random labels at the same budget. Policies compared at matched budgets:
      escalate-only      label a row only when the current readout is unsure (the first design)
      escalate + audit   also label a random eps share of all rows; the probe trains on both
      audit-train        the probe trains ONLY on the random audit rows; escalations just correct decisions
      typical-seed       first labels by typicality (k-means centres per question), then escalate + audit
    Stream = the train split in random order; held-out accuracy on the test split."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    L = args.layer; Xtr, Xte = D.X("train", L), D.X("test", L)
    tname = "gemma12b" if Path(f"{TEACHER12}.train.npy").exists() else "self"
    Ptr, Pte = D.teacher(tname, "train"), D.teacher(tname, "test")
    blend0 = 20
    seeds = {}
    for q in D.qkeys:   # typicality seeds: 5 per question, chosen without labels
        itr = D.idx_tr[q]; Z = PCA(n_components=32, random_state=0).fit_transform(Xtr[itr]); km = KMeans(n_clusters=5, n_init=3, random_state=0).fit(Z)
        seeds[q] = [int(itr[np.argmin(np.linalg.norm(Z - c, axis=1))]) for c in km.cluster_centers_]
    res = {"teacher": tname, "layer": L, "zero_shot_test_acc": acc_of(Pte, D.gte, D.kte), "runs": {}}

    def run(policy, tau, eps, seed=0):
        rng = np.random.default_rng(seed); order = rng.permutation(len(D.train))
        train_rows = {q: [] for q in D.qkeys}; model = {q: None for q in D.qkeys}; nfit = {q: 0 for q in D.qkeys}
        labels = 0; auto_ok = auto_n = 0
        if policy == "typical-seed":
            for q in D.qkeys:
                train_rows[q] = list(seeds[q]); labels += len(seeds[q])
                if len(np.unique(D.gtr[train_rows[q]])) >= 2:
                    model[q] = fit_lr(Xtr[train_rows[q]], D.gtr[train_rows[q]]); nfit[q] = len(train_rows[q])

        def probs(i, q, X, P0):
            k = D.kq[q]; p0 = P0[i, :k]
            if model[q] is None:
                return p0
            n = len(train_rows[q]); w = n / (n + blend0)
            return w * proba(model[q], X[i:i + 1], k)[0] + (1 - w) * p0

        for i in order:
            q = D.qtr[i]; p = probs(i, q, Xtr, Ptr)
            audit = rng.random() < eps
            unsure = p.max() < tau
            if audit or unsure:
                labels += 1
                if policy != "audit-train" or audit:
                    train_rows[q].append(i)
                    if len(train_rows[q]) - nfit[q] >= 5 and len(np.unique(D.gtr[train_rows[q]])) >= 2:
                        model[q] = fit_lr(Xtr[train_rows[q]], D.gtr[train_rows[q]]); nfit[q] = len(train_rows[q])
            else:
                auto_n += 1; auto_ok += int(np.argmax(p) == D.gtr[i])
        Pt = np.zeros((len(D.test), 26))
        for q in D.qkeys:
            ite = D.idx_te[q]; k = D.kq[q]
            if model[q] is None:
                Pt[ite, :k] = Pte[ite, :k]
            else:
                n = len(train_rows[q]); w = n / (n + blend0); Pt[ite, :k] = w * proba(model[q], Xte[ite], k) + (1 - w) * Pte[ite, :k]
        return dict(labels=labels, auto_rate=auto_n / len(order), auto_acc=auto_ok / max(1, auto_n), test_acc=acc_of(Pt, D.gte, D.kte))

    for tau in (0.6, 0.8, 0.9):
        for policy, eps in (("escalate-only", 0.0), ("escalate+audit", 0.05), ("escalate+audit", 0.10), ("audit-train", 0.10), ("typical-seed", 0.05)):
            key = f"tau{tau}_{policy}_eps{eps}"
            r = run(policy, tau, eps); res["runs"][key] = r
            # passive reference at the same budget
            ridx = np.random.default_rng(1).permutation(len(D.train))[: r["labels"]]
            lab = np.full(len(D.train), -1); lab[ridx] = D.gtr[ridx]
            Pp = probe_predict(D, L, lab, Xtr=Xtr, Xte=Xte); Pb = np.zeros_like(Pp)
            for q in D.qkeys:
                ite = D.idx_te[q]; n = int((lab[D.idx_tr[q]] >= 0).sum()); w = n / (n + blend0); k = D.kq[q]
                Pb[ite, :k] = w * Pp[ite, :k] + (1 - w) * Pte[ite, :k]
            r["random_same_budget_test_acc"] = acc_of(Pb, D.gte, D.kte)
            print(f"[{tname}] tau {tau} {policy:15s} eps {eps:.2f}: labels {r['labels']:5d} | auto {r['auto_rate']:.0%} at {r['auto_acc']:.3f} | "
                  f"held-out {r['test_acc']:.3f} vs random same budget {r['random_same_budget_test_acc']:.3f}", flush=True)
            Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ------------------------------------------------------------ few labels on top of a zero-shot prior
def exp_prior(D, args):
    """n labelled rows per question (random or most-typical), probe on layer L, combined with the 12B's
    zero-shot distribution: linear mixture w = n/(n+n0), or a product of experts (log-linear, weight 1)."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    L = args.layer; Xtr, Xte = D.X("train", L), D.X("test", L)
    Pz = D.teacher("gemma12b", "test")
    res = {"layer": L, "zero_shot_12b": acc_of(Pz, D.gte, D.kte), "n": {}}
    typ_cache = {}
    for n in (0, 5, 10, 25, 50):
        R = {}
        for sel in ("random", "typical"):
            accs = {k: [] for k in ("probe", "mix n0=10", "mix n0=20", "mix n0=40", "product of experts")}
            for seed in range(3):
                rng = np.random.default_rng(seed); Pp = np.zeros((len(D.test), 26))
                for q in D.qkeys:
                    itr, ite, k = D.idx_tr[q], D.idx_te[q], D.kq[q]
                    if n == 0:
                        Pp[ite, :k] = 1.0 / k; continue
                    if sel == "random":
                        pick = rng.choice(itr, n, replace=False)
                    else:
                        key = (q, n, seed)
                        if key not in typ_cache:
                            Z = PCA(n_components=32, random_state=seed).fit_transform(Xtr[itr]); km = KMeans(n_clusters=n, n_init=3, random_state=seed).fit(Z)
                            typ_cache[key] = np.array([itr[int(np.argmin(np.linalg.norm(Z - c, axis=1)))] for c in km.cluster_centers_])
                        pick = typ_cache[key]
                    y = D.gtr[pick]
                    if len(np.unique(y)) < 2:
                        Pp[ite, :k] = 0.1 / k; Pp[ite, int(y[0])] += 0.9; continue
                    Pp[ite, :k] = proba(fit_lr(Xtr[pick], y), Xte[ite], k)
                accs["probe"].append(acc_of(Pp, D.gte, D.kte) if n else float("nan"))
                for n0 in (10, 20, 40):
                    w = n / (n + n0); accs[f"mix n0={n0}"].append(acc_of(w * Pp + (1 - w) * Pz, D.gte, D.kte))
                poe = np.zeros_like(Pp)
                for i in range(len(D.test)):
                    k = D.kte[i]; poe[i, :k] = softmax(np.log(np.clip(Pp[i, :k], 1e-6, 1)) + np.log(np.clip(Pz[i, :k], 1e-6, 1)))
                accs["product of experts"].append(acc_of(poe, D.gte, D.kte))
            R[sel] = {k: float(np.nanmean(v)) if not all(np.isnan(v)) else None for k, v in accs.items()}
        res["n"][str(n)] = R
        print(f"n={n:2d}: " + " || ".join(f"{sel}: " + ", ".join(f"{k} {v:.3f}" for k, v in R[sel].items() if v is not None) for sel in R), flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ------------------------------------------------------------ is the decision code universal?
def exp_codetransfer(D, args):
    """PCA code fitted on the unlabelled states of three workflows, applied unchanged to the fourth.
    Per-question probes on the held-out workflow then use the transferred code (m numbers)."""
    from sklearn.decomposition import PCA
    L = args.layer; Xtr, Xte = D.X("train", L), D.X("test", L)
    wf_tr = np.array([q[0] for q in D.qtr])
    wfs = sorted(set(wf_tr))
    res = {"layer": L, "m": {}}
    for m in (8, 16, 32, 64, 128):
        own, transfer, full = [], [], []
        for w in wfs:
            qs = [q for q in D.qkeys if q[0] == w]
            p_own = PCA(n_components=m, random_state=0).fit(Xtr[wf_tr == w])
            p_tr = PCA(n_components=m, random_state=0).fit(Xtr[wf_tr != w])
            ok = {"own": 0, "transfer": 0, "full": 0}; n = 0
            for q in qs:
                itr, ite = D.idx_tr[q], D.idx_te[q]; y = D.gtr[itr]; yt = D.gte[ite]; n += len(ite)
                if len(np.unique(y)) < 2:
                    for k_ in ok:
                        ok[k_] += int((yt == y[0]).sum())
                    continue
                ok["own"] += int((fit_lr(p_own.transform(Xtr[itr]), y).predict(p_own.transform(Xte[ite])) == yt).sum())
                ok["transfer"] += int((fit_lr(p_tr.transform(Xtr[itr]), y).predict(p_tr.transform(Xte[ite])) == yt).sum())
                if m == 8:
                    ok["full"] += int((fit_lr(Xtr[itr], y).predict(Xte[ite]) == yt).sum())
            own.append(ok["own"] / n); transfer.append(ok["transfer"] / n); full.append(ok["full"] / n)
        res["m"][str(m)] = dict(code_from_same_workflow=float(np.mean(own)), code_from_other_workflows=float(np.mean(transfer)),
                                per_workflow_transfer={w: float(t) for w, t in zip(wfs, transfer)})
        if m == 8:
            res["full_features"] = float(np.mean(full))
        print(f"m={m:3d}: code fitted on the same workflow {np.mean(own):.3f} | on the other three {np.mean(transfer):.3f}", flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("full features (per-workflow mean):", res.get("full_features"), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["w2s", "anytime", "variants", "universal", "online", "lens", "labelmodel", "fewshot", "confsel", "stack", "typical", "intrinsic", "online2", "prior", "codetransfer"])
    ap.add_argument("--tag", default="qwen35-4b")
    ap.add_argument("--layers", default="22,26")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--grid", default="12,14,16,18,20,22,24,26,28")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    D = Data(args.tag)
    {"w2s": exp_w2s, "anytime": exp_anytime, "variants": exp_variants, "universal": exp_universal, "online": exp_online, "lens": exp_lens,
     "labelmodel": exp_labelmodel, "fewshot": exp_fewshot, "confsel": exp_confsel,
     "stack": exp_stack, "typical": exp_typical, "intrinsic": exp_intrinsic, "online2": exp_online2, "prior": exp_prior, "codetransfer": exp_codetransfer}[args.exp](D, args)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
