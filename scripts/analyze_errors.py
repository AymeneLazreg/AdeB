from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adeb.classify import classify
from adeb.evaluate import bootstrap_f1_ci, evaluate_once, stratified_split
from adeb.kb import KnowledgeBase
from adeb.learn import learn_greedy
from adeb.run import load_dataset
from adeb.signature import Config, build_signature


def main() -> None:
    kb = KnowledgeBase.from_json(Path("data/jdm_complet.json"))
    items = load_dataset(Path("data/corpus_complet.tsv"), kb)
    cfg = Config(use_def=True, trim=True, fusion_threshold=0.50, classify_agg="mean")

    train_items, test_items = stratified_split(items, train_ratio=50 / 80, seed=0)
    metrics, y_true, y_pred = evaluate_once(kb, train_items, test_items, cfg)
    lo, hi = bootstrap_f1_ci(y_true, y_pred, n_resamples=1000, seed=42)

    print(f"Split seed=0 : {len(train_items)} train / {len(test_items)} test | Règles={metrics.n_rules}")
    print(f"Accuracy={metrics.accuracy:.3f} | Macro-P={metrics.macro_p:.3f} | Macro-R={metrics.macro_r:.3f}")
    print(f"Macro-F1={metrics.macro_f1:.3f} (IC 95% bootstrap : [{lo:.3f}, {hi:.3f}])\n")

    print(f"{'Classe':6s} | {'Support':7s} | {'Précision':9s} | {'Rappel':7s} | {'F1':7s}")
    print("-" * 50)
    for rt, d in sorted(metrics.per_class.items(), key=lambda x: x[1]["f1"]):
        print(
            f"{rt:6s} | {int(d['support']):7d} | "
            f"{d['precision']:9.3f} | {d['recall']:7.3f} | {d['f1']:7.3f}"
        )

    errors_by_pair: Counter[tuple[str, str]] = Counter(
        (g, p) for g, p in zip(y_true, y_pred) if g != p
    )
    print("\nTop 10 paires de confusion (or -> prédit) :")
    for (gold, pred), cnt in errors_by_pair.most_common(10):
        print(f"  {gold:4s} -> {pred:4s} : {cnt}")

    train_vecs = [
        (
            build_signature(kb, p.a, cfg, p.feats, "A"),
            build_signature(kb, p.b, cfg, p.feats, "B"),
            rt,
        )
        for p, rt in train_items
    ]
    rules = learn_greedy(train_vecs, cfg)

    print("\nExemples d'erreurs explicables (6 premiers cas) :")
    shown = 0
    for (parsed, gold), pred in zip(test_items, y_pred):
        if gold == pred:
            continue
        sa = build_signature(kb, parsed.a, cfg, parsed.feats, "A")
        sb = build_signature(kb, parsed.b, cfg, parsed.feats, "B")
        rank, exp = classify(rules, sa, sb, cfg, explain=True)
        top2 = [(rt, round(sc, 3)) for sc, rt in rank[:2]]
        print(
            f"  - '{parsed.raw}' (A='{parsed.a}', B='{parsed.b}') | "
            f"or={gold} -> prédit={pred} | top2={top2}"
        )
        if exp:
            print(
                f"      sim_A={exp.sim_a:.3f}, sim_B={exp.sim_b:.3f}, w={exp.rule_weight} | "
                f"A:{[k for k, _ in exp.top_a_symbols[:2]]} | B:{[k for k, _ in exp.top_b_symbols[:2]]}"
            )
        shown += 1
        if shown >= 6:
            break


if __name__ == "__main__":
    main()