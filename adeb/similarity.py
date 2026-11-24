from __future__ import annotations

import math

from adeb.signature import Vec


def norm2(u: Vec) -> float:
    """Norme euclidienne L2 d'un vecteur creux."""
    return math.sqrt(sum(w * w for w in u.values()))


def dot_product(u: Vec, v: Vec) -> float:
    """Produit scalaire creux en O(min(|u|, |v|)) en temps moyen."""
    if not u or not v:
        return 0.0
    if len(u) > len(v):
        u, v = v, u
    return sum(w * v[k] for k, w in u.items() if k in v)


def cosine(u: Vec, v: Vec) -> float:
    """Similarité cosinus entre deux vecteurs creux."""
    if not u or not v:
        return 0.0
    nu = norm2(u)
    nv = norm2(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot_product(u, v) / (nu * nv)


def vsum(u: Vec, v: Vec) -> Vec:
    """Somme vectorielle (= union pondérée de deux signatures)."""
    out = dict(u)
    for k, w in v.items():
        out[k] = out.get(k, 0.0) + w
    return out


def aggregate_scores(sim_a: float, sim_b: float, mode: str = "mean") -> float:
    """Combine les similarités côté A et côté B."""
    if mode == "mean":
        return 0.5 * (sim_a + sim_b)
    if mode == "min":
        return min(sim_a, sim_b)
    if mode == "product":
        return sim_a * sim_b
    raise ValueError(f"Mode d'agrégation inconnu : {mode}")