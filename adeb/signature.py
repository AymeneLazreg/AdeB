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
    hyper_depth: int = 2
    hyper_min_weight: float = 0.0
    trt_min_weight: float = 0.0
    trt_log_scale: bool = True       # Atténue les relations massives (r_lieu, r_holo)
    normalize_blocks: bool = True    # Chaque bloc activé a une norme L2 égale à 1
    fusion_threshold: float = 0.5
    trim: bool = False
    fusion_agg: str = "mean"
    classify_agg: str = "mean"


def _l2_normalize(block: Vec) -> Vec:
    norm = math.sqrt(sum(w * w for w in block.values()))
    if norm == 0.0:
        return block
    return {k: w / norm for k, w in block.items()}


def build_signature(
    kb: KnowledgeBase,
    term: str,
    cfg: Config,
    feats: dict[str, bool] | None = None,
    side: str = "A",
) -> Vec:
    """Construit le vecteur creux s(term) avec normalisation par bloc."""
    blocks: list[Vec] = []

    if cfg.use_h:
        b_h: Vec = {f"H:{term}": 1.0}
        for hyper, w in kb.hypernyms(
            term, depth=cfg.hyper_depth, min_weight=cfg.hyper_min_weight
        ).items():
            b_h[f"H:{hyper}"] = float(w)
        blocks.append(b_h)

    if cfg.use_trt:
        b_t: Vec = {}
        for rel, count in kb.incoming_rel_types(
            term, min_weight=cfg.trt_min_weight
        ).items():
            b_t[f"T:{rel}"] = math.log1p(count) if cfg.trt_log_scale else float(count)
        if b_t:
            blocks.append(b_t)

    if cfg.use_sst:
        b_s: Vec = {f"S:{st}": 1.0 for st in kb.sem_types(term)}
        if b_s:
            blocks.append(b_s)

    if cfg.use_def and side == "B" and feats is not None:
        b_d: Vec = {
            "D:Det" if feats["Det"] else "D:NoDet": 1.0,
            "D:Def" if feats["Def"] else "D:NoDef": 1.0,
        }
        blocks.append(b_d)

    vec: Vec = {}
    for block in blocks:
        norm_block = _l2_normalize(block) if cfg.normalize_blocks else block
        vec.update(norm_block)
    return vec