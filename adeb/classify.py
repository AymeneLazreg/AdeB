from __future__ import annotations

from dataclasses import dataclass

from adeb.learn import Rule
from adeb.signature import Config, Vec
from adeb.similarity import aggregate_scores, cosine


@dataclass
class Explanation:
    predicted_type: str
    score: float
    rule_weight: int
    sim_a: float
    sim_b: float
    top_a_symbols: list[tuple[str, float]]
    top_b_symbols: list[tuple[str, float]]


def classify(
    rules: list[Rule],
    sa: Vec,
    sb: Vec,
    cfg: Config,
    explain: bool = False,
    top_symbols: int = 5,
) -> list[tuple[float, str]] | tuple[list[tuple[float, str]], Explanation | None]:
    """
    Classe (s(A), s(B)) en retenant le meilleur score par type de relation.
    Retourne le classement décroissant [(score, type), ...], et une Explanation si explain=True.
    """
    if not rules:
        return ([], None) if explain else []

    best_per_type: dict[str, tuple[float, float, float, Rule]] = {}

    for r in rules:
        sim_a = cosine(sa, r.sl)
        sim_b = cosine(sb, r.sr)
        score = aggregate_scores(sim_a, sim_b, mode=cfg.classify_agg)
        if r.rt not in best_per_type or score > best_per_type[r.rt][0]:
            best_per_type[r.rt] = (score, sim_a, sim_b, r)

    ranking = sorted(
        ((vals[0], rt) for rt, vals in best_per_type.items()),
        key=lambda x: x[0],
        reverse=True,
    )

    if not explain:
        return ranking

    top_score, top_rt = ranking[0]
    _, sim_a, sim_b, top_rule = best_per_type[top_rt]

    contrib_a = sorted(
        ((k, sa[k] * top_rule.sl[k]) for k in (sa.keys() & top_rule.sl.keys())),
        key=lambda x: x[1],
        reverse=True,
    )[:top_symbols]
    contrib_b = sorted(
        ((k, sb[k] * top_rule.sr[k]) for k in (sb.keys() & top_rule.sr.keys())),
        key=lambda x: x[1],
        reverse=True,
    )[:top_symbols]

    exp = Explanation(
        predicted_type=top_rt,
        score=top_score,
        rule_weight=top_rule.weight,
        sim_a=sim_a,
        sim_b=sim_b,
        top_a_symbols=contrib_a,
        top_b_symbols=contrib_b,
    )
    return ranking, exp