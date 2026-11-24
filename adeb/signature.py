from __future__ import annotations

import math
from dataclasses import dataclass

from adeb.kb import KnowledgeBase

Vec = dict[str, float]  # vecteur creux : symbole -> poids


@dataclass(frozen=True)
class Config:
    use_h: bool = True
    use_trt: bool = True
    use_sst: bool = True
    use_def: bool = False
    hyper_depth: int = 3
    hyper_min_weight: float = 0.0
    trt_min_weight: float = 0.0
    trt_log_scale: bool = False
    fusion_threshold: float = 0.5
    trim: bool = False
    fusion_agg: str = "mean"    # [Choix] "mean" ou "both"
    classify_agg: str = "mean"  # [Choix] "mean", "min", "product"


def build_signature(
    kb: KnowledgeBase,
    term: str,
    cfg: Config,
    feats: dict[str, bool] | None = None,
    side: str = "A",
) -> Vec:
    """Construit le vecteur creux s(term) selon les blocs activés dans `cfg`."""
    vec: Vec = {}

    if cfg.use_h:
        # Le terme lui-même est inclus pour capter ses hyponymes
        vec[f"H:{term}"] = 1.0
        for hyper, w in kb.hypernyms(
            term, depth=cfg.hyper_depth, min_weight=cfg.hyper_min_weight
        ).items():
            vec[f"H:{hyper}"] = float(w)

    if cfg.use_trt:
        for rel, count in kb.incoming_rel_types(
            term, min_weight=cfg.trt_min_weight
        ).items():
            val = math.log1p(count) if cfg.trt_log_scale else float(count)
            vec[f"T:{rel}"] = val

    if cfg.use_sst:
        for st in kb.sem_types(term):
            vec[f"S:{st}"] = 1.0

    if cfg.use_def and side == "B" and feats is not None:
        vec["D:Det" if feats["Det"] else "D:NoDet"] = 1.0
        vec["D:Def" if feats["Def"] else "D:NoDef"] = 1.0

    return vec