
import numpy as np

CODES = [f"A{i:02}" for i in range(1, 9)]


def matrix(predictions):
    cm = np.zeros((8, 9), dtype=np.int64)
    for p in predictions:
        if p["reference"] not in CODES:
            raise ValueError("Unknown reference label")
        cm[
            CODES.index(p["reference"]),
            CODES.index(p["prediction"]) if p["prediction"] in CODES else 8,
        ] += 1
    return cm


def scores(cm):
    cm = np.asarray(cm)
    tp = np.diagonal(cm[..., :8], axis1=-2, axis2=-1)
    support = cm.sum(-1)
    predicted = cm[..., :8].sum(-2)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0)
    return np.stack((tp.sum(-1) / cm.sum(axis=(-2, -1)), f1.mean(-1)), axis=-1)


def metrics(predictions):
    cm = matrix(predictions)
    acc, f1 = scores(cm)
    per = {}
    for i, c in enumerate(CODES):
        tp = int(cm[i, i])
        support = int(cm[i].sum())
        pred = int(cm[:, i].sum())
        per[c] = {
            "precision": tp / pred if pred else 0.0,
            "recall": tp / support if support else 0.0,
            "f1": 2 * tp / (support + pred) if support + pred else 0.0,
            "support": support,
        }
    return {
        "examples": len(predictions),
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "per_class": per,
        "predicted_class_frequencies": {
            c: int(cm[:, i].sum()) for i, c in enumerate(CODES)
        },
        "invalid_outputs": int(cm[:, 8].sum()),
        "confusion_matrix": {
            "rows": CODES,
            "columns": CODES + ["INVALID"],
            "counts": cm.tolist(),
        },
        "metric_convention": "All eight actions; precision/recall/F1 zero when their denominator is zero; invalid predictions count as incorrect and as false negatives.",
    }


def paired_bootstrap(confusions, repetitions=10000, seed=20261027):
    """Input [size(9B,2B),condition(none,auto),seed,conversation,ref,pred]."""
    a = np.asarray(confusions)
    n = a.shape[3]
    totals = a.sum(axis=3)
    point = scores(totals)
    effect = (point[:, 1] - point[:, 0]).mean(axis=1)
    contrast = effect[1] - effect[0]
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, repetitions, 64):
        # One shared multiplicity vector for every size, condition and seed.
        w = rng.multinomial(n, np.full(n, 1 / n), size=min(64, repetitions - start))
        boot = np.einsum("bc,xyzcrp->bxyzrp", w, a, optimize=True)
        vals = scores(boot)
        eff = (vals[:, :, 1] - vals[:, :, 0]).mean(axis=2)
        samples.append(
            np.concatenate((eff, (eff[:, 1] - eff[:, 0])[:, None, :]), axis=1)
        )
    dist = np.concatenate(samples)
    ci = np.quantile(dist, [0.025, 0.975], axis=0)
    result = {
        "bootstrap_seed": seed,
        "repetitions": repetitions,
        "interval_method": "95% percentile, numpy linear quantiles",
        "unit": "conversation, sampled with replacement; same multiplicities across all twelve runs",
        "scope": "Conditional on the trained runs; does not fully capture retraining uncertainty.",
        "units": "percentage points",
        "metrics": {},
    }
    for i, name in enumerate(("accuracy", "macro_f1")):
        result["metrics"][name] = {}
        for j, key in enumerate(("effect_9B", "effect_2B", "contrast_2B_minus_9B")):
            val = effect[j, i] if j < 2 else contrast[i]
            result["metrics"][name][key] = {
                "estimate": 100 * float(val),
                "ci95": (100 * ci[:, j, i]).tolist(),
            }
    return result
