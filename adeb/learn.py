from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from adeb.signature import Config, Vec
from adeb.similarity import cosine, vsum


@dataclass
class Rule:
    sl: Vec
    sr: Vec
    rt: str
    weight: int = 0     # Nombre de fusions ayant produit la règle
    used: bool = False  # True si absorbée dans une fusion (élaguée si Trim=True)


def merge_rules(r1: Rule, r2: Rule) -> Rule:
    """Fusionne deux règles de même type par union pondérée (somme vectorielle)."""
    if r1.rt != r2.rt:
        raise ValueError(f"Fusion interdite entre types différents : {r1.rt} vs {r2.rt}")
    return Rule(
        sl=vsum(r1.sl, r2.sl),
        sr=vsum(r1.sr, r2.sr),
        rt=r1.rt,
        weight=r1.weight + r2.weight + 1,
        used=False,
    )


def rules_can_merge(r1: Rule, r2: Rule, cfg: Config) -> tuple[bool, float]:
    """
    Vérifie si deux règles sont fusionnables selon le critère choisi [Choix] :
    - 'mean' : moyenne des cosinus gauche/droite > seuil
    - 'both' : cosinus gauche > seuil ET cosinus droit > seuil
    """
    sim_l = cosine(r1.sl, r2.sl)
    sim_r = cosine(r1.sr, r2.sr)
    score = 0.5 * (sim_l + sim_r)
    if cfg.fusion_agg == "both":
        return (sim_l > cfg.fusion_threshold and sim_r > cfg.fusion_threshold), score
    return (score > cfg.fusion_threshold), score


def learn_greedy(
    examples: list[tuple[Vec, Vec, str]],
    cfg: Config,
) -> list[Rule]:
    """
    Apprentissage glouton séquentiel (socle GRASP-it).
    Chaque exemple est comparé aux règles encore actives du même type ;
    s'il en existe une assez proche, on crée une nouvelle règle fusionnée.
    """
    by_type: dict[str, list[Rule]] = defaultdict(list)

    for sl, sr, rt in examples:
        new_rule = Rule(sl=dict(sl), sr=dict(sr), rt=rt)
        by_type[rt].append(new_rule)

        best_rule: Rule | None = None
        best_score = -1.0

        for cand in by_type[rt][:-1]:
            if cand.used:
                continue
            ok, score = rules_can_merge(cand, new_rule, cfg)
            if ok and score > best_score:
                best_rule = cand
                best_score = score

        if best_rule is not None:
            fused = merge_rules(best_rule, new_rule)
            best_rule.used = True
            new_rule.used = True
            by_type[rt].append(fused)

    all_rules = [r for group in by_type.values() for r in group]
    if cfg.trim:
        return [r for r in all_rules if not r.used]
    return all_rules