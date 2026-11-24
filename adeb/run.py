from __future__ import annotations

import argparse
import csv
from pathlib import Path

from adeb.classify import classify
from adeb.evaluate import evaluate_multi_seed
from adeb.kb import KnowledgeBase
from adeb.learn import learn_greedy
from adeb.preprocess import ParsedPhrase, preprocess_phrase
from adeb.signature import Config, build_signature


def load_dataset(
    corpus_path: Path, kb: KnowledgeBase
) -> list[tuple[ParsedPhrase, str]]:
    """Charge un corpus TSV (colonnes 'phrase' et 'type') filtré sur la KB."""
    items: list[tuple[ParsedPhrase, str]] = []
    with open(corpus_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            parsed = preprocess_phrase(row["phrase"], kb=kb)
            if parsed and kb.exists(parsed.a) and kb.exists(parsed.b):
                items.append((parsed, row["type"]))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Socle GRASP-it — TP A de B")
    parser.add_argument("--kb", type=Path, default=Path("data/mini_kb.json"))
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus_exemple.tsv"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--demo", type=str, default="désert d'Algérie")
    args = parser.parse_args()

    kb = KnowledgeBase.from_json(args.kb)
    items = load_dataset(args.corpus, kb)
    print(f"KB chargée ({len(kb.terms)} termes) | {len(items)} exemples exploitables.\n")

    experiments: dict[str, Config] = {
        "Exp1: H seul": Config(use_h=True, use_trt=False, use_sst=False, use_def=False),
        "Exp1: H+SST": Config(use_h=True, use_trt=False, use_sst=True, use_def=False),
        "Exp1: H+TRT": Config(use_h=True, use_trt=True, use_sst=False, use_def=False),
        "Exp1: H+TRT+SST": Config(use_h=True, use_trt=True, use_sst=True, use_def=False),
        "Exp2: H+TRT+SST+DEF": Config(use_h=True, use_trt=True, use_sst=True, use_def=True),
        "Exp3: +DEF (Trim)": Config(use_h=True, use_trt=True, use_sst=True, use_def=True, trim=True),
    }

    seeds = tuple(range(args.seeds))
    print(f"{'Expérience':22s} | {'F1 macro':15s} | {'Règles':7s} | {'Temps test (ms)':15s}")
    print("-" * 68)
    for name, cfg in experiments.items():
        res = evaluate_multi_seed(kb, items, cfg, seeds=seeds)
        print(
            f"{name:22s} | "
            f"{res['macro_f1_mean']:.3f} ± {res['macro_f1_std']:.3f}   | "
            f"{res['n_rules_mean']:7.1f} | "
            f"{res['classify_ms_mean']:10.2f}"
        )

    print("\n--- Démonstration explicable ---")
    cfg_demo = Config(use_def=True)
    train_vecs = [
        (
            build_signature(kb, p.a, cfg_demo, p.feats, "A"),
            build_signature(kb, p.b, cfg_demo, p.feats, "B"),
            rt,
        )
        for p, rt in items
    ]
    rules = learn_greedy(train_vecs, cfg_demo)
    parsed_demo = preprocess_phrase(args.demo, kb=kb)
    if parsed_demo:
        sa = build_signature(kb, parsed_demo.a, cfg_demo, parsed_demo.feats, "A")
        sb = build_signature(kb, parsed_demo.b, cfg_demo, parsed_demo.feats, "B")
        ranking, exp = classify(rules, sa, sb, cfg_demo, explain=True)
        print(
            f"Entrée : '{args.demo}' -> A='{parsed_demo.a}', "
            f"B='{parsed_demo.b}', feats={parsed_demo.feats}"
        )
        print("Top 3  :", [(rt, round(sc, 3)) for sc, rt in ranking[:3]])
        if exp:
            print(
                f"Règle gagnante (w={exp.rule_weight}) | "
                f"sim_A={exp.sim_a:.3f}, sim_B={exp.sim_b:.3f}"
            )
            print("  Symboles contributifs A :", exp.top_a_symbols)
            print("  Symboles contributifs B :", exp.top_b_symbols)


if __name__ == "__main__":
    main()