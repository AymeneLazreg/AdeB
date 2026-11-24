from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from adeb.kb import KnowledgeBase

# Les formes composées ("de la ", "de l'") doivent apparaître avant "de " et "d'"
PREP_PATTERN = re.compile(r"\s+(de la |de l'|du |des |de |d')", flags=re.IGNORECASE)

INDEF_DETS = ("un ", "une ", "des ", "quelques ", "plusieurs ", "certains ", "certaines ")
DEF_DETS = (
    "le ", "la ", "les ", "l'",
    "mon ", "ma ", "mes ", "ton ", "ta ", "tes ",
    "son ", "sa ", "ses ", "notre ", "nos ", "votre ", "vos ", "leur ", "leurs ",
    "ce ", "cet ", "cette ", "ces ",
)


@dataclass(frozen=True)
class ParsedPhrase:
    raw: str
    a: str
    b: str
    prep: str
    feats: dict[str, bool]  # {"Det": bool, "Def": bool}


def simple_french_lemma(term: str, kb: KnowledgeBase | None = None) -> str:
    """
    Lemmatisation simple sans dépendance externe.
    Si le terme exact est déjà dans la KB, on le conserve tel quel.
    Sinon on teste les formes singulières usuelles du français.
    """
    t = term.strip()
    if kb and kb.exists(t):
        return t

    candidates: list[str] = []
    if t.endswith("aux") and len(t) > 4:
        candidates.append(t[:-3] + "al")
    if t.endswith("eaux") and len(t) > 4:
        candidates.append(t[:-1])
    if t.endswith("s") or t.endswith("x"):
        candidates.append(t[:-1])
    if t and t[0].isupper():
        candidates.append(t.lower())

    if kb:
        for cand in candidates:
            if kb.exists(cand):
                return cand
    return candidates[0] if (candidates and kb is None) else t


def strip_leading_det(text: str) -> tuple[str, bool, bool]:
    """
    Retire un éventuel déterminant au début d'un groupe nominal.
    Retourne (texte_nettoyé, a_un_determinant, est_defini).
    """
    low = text.lower()
    for d in DEF_DETS:
        if low.startswith(d):
            return text[len(d):].strip(), True, True
    for d in INDEF_DETS:
        if low.startswith(d):
            return text[len(d):].strip(), True, False
    return text.strip(), False, False


def generate_split_candidates(phrase: str) -> list[ParsedPhrase]:
    """Génère tous les découpages (A, B) possibles sur chaque préposition 'de/du/des/d'/...'."""
    s = phrase.strip()
    candidates: list[ParsedPhrase] = []

    for m in PREP_PATTERN.finditer(s):
        raw_a = s[: m.start()].strip()
        prep = m.group(1)
        raw_b = s[m.end() :].strip()
        if not raw_a or not raw_b:
            continue

        clean_a, _, _ = strip_leading_det(raw_a)
        clean_b, det_in_b, def_in_b = strip_leading_det(raw_b)

        p_norm = prep.strip().lower()
        # 'du', 'des', 'de la', 'de l'' portent déjà un déterminant contracté
        det = p_norm in ("du", "des", "de la", "de l'") or det_in_b
        # Nom propre (majuscule initiale) forcé à Def=True
        is_proper = bool(clean_b) and clean_b[0].isupper()
        definite = p_norm in ("du", "de la", "de l'") or def_in_b or is_proper

        candidates.append(
            ParsedPhrase(
                raw=s,
                a=clean_a,
                b=clean_b,
                prep=prep.strip(),
                feats={"Det": det, "Def": definite},
            )
        )
    return candidates


def preprocess_phrase(
    phrase: str,
    kb: KnowledgeBase | None = None,
    ne_fallback: Callable[[str], str] | None = None,
    lemmatize: bool = True,
) -> ParsedPhrase | None:
    """
    Segmente et normalise une expression 'A de B'.
    Pour les prépositions multiples, privilégie le candidat dont A et B existent dans la KB.
    """
    candidates = generate_split_candidates(phrase)
    if not candidates:
        return None

    def normalize(cand: ParsedPhrase) -> ParsedPhrase:
        a, b = cand.a, cand.b
        if lemmatize:
            a = simple_french_lemma(a, kb)
            b = simple_french_lemma(b, kb)
        if kb and ne_fallback:
            if not kb.exists(a) and a[:1].isupper():
                a = ne_fallback(a)
            if not kb.exists(b) and b[:1].isupper():
                b = ne_fallback(b)
        return ParsedPhrase(cand.raw, a, b, cand.prep, cand.feats)

    normalized = [normalize(c) for c in candidates]
    if kb is None or len(normalized) == 1:
        return normalized[0]

    # Priorité 1 : A et B existent tous les deux dans la KB
    for cand in normalized:
        if kb.exists(cand.a) and kb.exists(cand.b):
            return cand

    # Priorité 2 (heuristique) : au moins un des deux termes existe dans la KB
    for cand in normalized:
        if kb.exists(cand.a) or kb.exists(cand.b):
            return cand

    return normalized[0]