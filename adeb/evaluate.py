from __future__ import annotations

import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from adeb.classify import classify
from adeb.kb import KnowledgeBase
from adeb.learn import learn_greedy
from adeb.preprocess import ParsedPhrase
from adeb.signature import Config, build_signature


@dataclass
class EvalMetrics:
    accuracy: float
    macro_p: float
    macro_r: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: dict[tuple[str, str], int]
    n_rules: int
    classify_seconds: float


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    n_rules: int = 0,
    classify_seconds: float = 0.0,
) -> EvalMetrics:
    """Calcule précision, rappel, F1 par classe, moyennes macro et matrice de confusion."""
    labels = sorted(set(y_true) | set(y_pred))
    confusion: dict[tuple[str, str], int] = Counter(zip(y_true, y_pred))

    per_class: dict[str, dict[str, float]] = {}
    for c in labels:
        tp = confusion[(c, c)]
        fp = sum(confusion[(g, c)] for g in labels if g != c)
        fn = sum(confusion[(c, p)] for p in labels if p != c)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_class[c] = {"precision": p, "recall": r, "f1": f1, "support": tp + fn}

    gold_labels = sorted(set(y_true))
    macro_p = sum(per_class[c]["precision"] for c in gold_labels) / len(gold_labels)
    macro_r = sum(per_class[c]["recall"] for c in gold_labels) / len(gold_labels)
    macro_f1 = sum(per_class[c]["f1"] for c in gold_labels) / len(gold_labels)
    acc = sum(1 for g, p in zip(y_true, y_pred) if g == p) / len(y_true)

    return EvalMetrics(
        accuracy=acc,
        macro_p=macro_p,
        macro_r=macro_r,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion=dict(confusion),
        n_rules=n_rules,
        classify_seconds=classify_seconds,
    )


def bootstrap_f1_ci(
    y_true: list[str],
    y_pred: list[str],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Intervalle de confiance (1 - alpha) par bootstrap sur le F1 macro."""
    rng = random.Random(seed)
    n = len(y_true)
    scores: list[float] = []
    for _ in range(n_resamples):
        idxs = [rng.randrange(n) for _ in range(n)]
        yt = [y_true[i] for i in idxs]
        yp = [y_pred[i] for i in idxs]
        scores.append(compute_metrics(yt, yp).macro_f1)
    scores.sort()
    lo = scores[int((alpha / 2) * n_resamples)]
    hi = scores[int((1 - alpha / 2) * n_resamples) - 1]
    return lo, hi


def stratified_split(
    items: list[tuple[ParsedPhrase, str]],
    train_ratio: float = 50 / 80,
    seed: int = 0,
) -> tuple[list[tuple[ParsedPhrase, str]], list[tuple[ParsedPhrase, str]]]:
    """Découpage stratifié train/test strictement séparé par classe."""
    rng = random.Random(seed)
    by_label: dict[str, list[tuple[ParsedPhrase, str]]] = defaultdict(list)
    for item in items:
        by_label[item[1]].append(item)

    train: list[tuple[ParsedPhrase, str]] = []
    test: list[tuple[ParsedPhrase, str]] = []
    for _label, group in sorted(by_label.items()):
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_train = max(1, int(round(len(shuffled) * train_ratio)))
        if n_train >= len(shuffled) and len(shuffled) > 1:
            n_train = len(shuffled) - 1
        train.extend(shuffled[:n_train])
        test.extend(shuffled[n_train:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def evaluate_once(
    kb: KnowledgeBase,
    train_items: list[tuple[ParsedPhrase, str]],
    test_items: list[tuple[ParsedPhrase, str]],
    cfg: Config,
) -> tuple[EvalMetrics, list[str], list[str]]:
    """Entraîne sur `train_items` et évalue sur `test_items`."""
    train_vecs = [
        (
            build_signature(kb, p.a, cfg, p.feats, "A"),
            build_signature(kb, p.b, cfg, p.feats, "B"),
            rt,
        )
        for p, rt in train_items
    ]
    rules = learn_greedy(train_vecs, cfg)

    y_true: list[str] = []
    y_pred: list[str] = []
    t0 = time.perf_counter()
    for p, gold in test_items:
        sa = build_signature(kb, p.a, cfg, p.feats, "A")
        sb = build_signature(kb, p.b, cfg, p.feats, "B")
        ranking = classify(rules, sa, sb, cfg, explain=False)
        pred = ranking[0][1] if ranking else "UNKNOWN"
        y_true.append(gold)
        y_pred.append(pred)
    elapsed = time.perf_counter() - t0

    metrics = compute_metrics(
        y_true, y_pred, n_rules=len(rules), classify_seconds=elapsed
    )
    return metrics, y_true, y_pred


def evaluate_multi_seed(
    kb: KnowledgeBase,
    items: list[tuple[ParsedPhrase, str]],
    cfg: Config,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    train_ratio: float = 50 / 80,
) -> dict[str, float]:
    """Répète train/test sur plusieurs graines et renvoie moyenne ± écart-type."""
    f1s, accs, n_rules_list, times = [], [], [], []
    for s in seeds:
        train_set, test_set = stratified_split(items, train_ratio=train_ratio, seed=s)
        metrics, _, _ = evaluate_once(kb, train_set, test_set, cfg)
        f1s.append(metrics.macro_f1)
        accs.append(metrics.accuracy)
        n_rules_list.append(metrics.n_rules)
        times.append(metrics.classify_seconds)

    def mean_std(vals: list[float]) -> tuple[float, float]:
        m = sum(vals) / len(vals)
        var = sum((x - m) ** 2 for x in vals) / len(vals)
        return m, math.sqrt(var)

    f1_m, f1_s = mean_std(f1s)
    acc_m, acc_s = mean_std(accs)
    r_m, _ = mean_std(n_rules_list)
    t_m, _ = mean_std(times)
    return {
        "macro_f1_mean": f1_m,
        "macro_f1_std": f1_s,
        "acc_mean": acc_m,
        "acc_std": acc_s,
        "n_rules_mean": r_m,
        "classify_ms_mean": t_m * 1000.0,
    }