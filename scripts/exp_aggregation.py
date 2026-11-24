from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adeb.evaluate import evaluate_multi_seed
from adeb.kb import KnowledgeBase
from adeb.run import load_dataset
from adeb.signature import Config


def main() -> None:
    kb = KnowledgeBase.from_json(Path("data/jdm_complet.json"))
    items = load_dataset(Path("data/corpus_complet.tsv"), kb)
    seeds = (0, 1, 2, 3, 4)

    print(f"Comparaison agrégation A/B sur {len(items)} exemples (Trim=True, +DEF)\n")
    print(f"{'Agrégation':12s} | {'F1 macro':15s} | {'Accuracy':15s} | {'Règles':7s} | {'Temps (ms)':10s}")
    print("-" * 74)

    for mode in ("mean", "min", "product"):
        cfg = Config(
            use_h=True,
            use_trt=True,
            use_sst=True,
            use_def=True,
            trim=True,
            classify_agg=mode,
        )
        res = evaluate_multi_seed(kb, items, cfg, seeds=seeds)
        print(
            f"{mode:12s} | "
            f"{res['macro_f1_mean']:.3f} ± {res['macro_f1_std']:.3f}   | "
            f"{res['acc_mean']:.3f} ± {res['acc_std']:.3f}   | "
            f"{res['n_rules_mean']:7.1f} | "
            f"{res['classify_ms_mean']:10.2f}"
        )


if __name__ == "__main__":
    main()