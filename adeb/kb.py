from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


class KnowledgeBase:
    """Graphe lexico-sémantique orienté, pondéré et typé."""

    def __init__(
        self,
        relations: Iterable[tuple[str, str, str, float]],
        sem_types: dict[str, list[str]],
        trt_counts: dict[str, dict[str, int]] | None = None,
    ) -> None:
        self.out: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        self.inc: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        self.sem: dict[str, list[str]] = defaultdict(list)
        self.trt_counts: dict[str, dict[str, int]] = trt_counts or {}
        self.refinements: dict[str, list[str]] = defaultdict(list)

        for src, rel, dst, weight in relations:
            self.out[src].append((rel, dst, float(weight)))
            self.inc[dst].append((rel, src, float(weight)))

        for term, types in sem_types.items():
            self.sem[term] = list(types)

        self.terms: set[str] = (
            set(self.out) | set(self.inc) | set(self.sem) | set(self.trt_counts)
        )

        # Repère les raffinements de sens (ex: "avocat>justice" rattaché à "avocat")
        for term in self.terms:
            if ">" in term:
                base = term.split(">", 1)[0]
                self.refinements[base].append(term)

    @classmethod
    def from_json(cls, path: str | Path) -> KnowledgeBase:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            data["relations"],
            data.get("sem_types", {}),
            trt_counts=data.get("trt_counts"),
        )

    def exists(self, term: str) -> bool:
        return term in self.terms

    def _expand_senses(self, term: str, include_refinements: bool = True) -> list[str]:
        """Socle : union du terme principal et de ses raffinements de sens."""
        nodes = [term] if term in self.terms else []
        if include_refinements and term in self.refinements:
            nodes.extend(self.refinements[term])
        return nodes

    def neighbors(
        self,
        term: str,
        rel: str | None = None,
        direction: str = "out",
        min_weight: float = 0.0,
    ) -> list[tuple[str, str, float]]:
        """Retourne les arcs sortants ('out') ou entrants ('inc') d'un terme."""
        adj = self.out if direction == "out" else self.inc
        return [
            (r, target, w)
            for r, target, w in adj.get(term, [])
            if (rel is None or r == rel) and w >= min_weight
        ]

    def hypernyms(
        self,
        term: str,
        depth: int = 3,
        min_weight: float = 0.0,
        decay: float = 0.7,
        include_refinements: bool = True,
    ) -> Counter[str]:
        """Parcours BFS sur r_isa jusqu'à `depth` niveaux."""
        res: Counter[str] = Counter()
        start_nodes = self._expand_senses(term, include_refinements)
        frontier: list[tuple[str, float]] = [(n, 1.0) for n in start_nodes]
        best_seen: dict[str, float] = {n: 1.0 for n in start_nodes}

        for _ in range(depth):
            nxt: list[tuple[str, float]] = []
            for curr, w0 in frontier:
                for rel, dst, w in self.out.get(curr, []):
                    if rel == "r_isa" and w >= min_weight:
                        score = w0 * (w / 100.0)
                        if score > res[dst]:
                            res[dst] = score
                        next_w0 = w0 * decay
                        if next_w0 > best_seen.get(dst, 0.0):
                            best_seen[dst] = next_w0
                            nxt.append((dst, next_w0))
            frontier = nxt
            if not frontier:
                break
        return res

    def incoming_rel_types(
        self,
        term: str,
        min_weight: float = 0.0,
        include_refinements: bool = True,
    ) -> Counter[str]:
        """Compte les types de relations entrantes (TRT) pointant vers `term`."""
        counts: Counter[str] = Counter()
        for node in self._expand_senses(term, include_refinements):
            if node in self.trt_counts:
                for rel, cnt in self.trt_counts[node].items():
                    if rel != "r_isa":
                        counts[rel] += cnt
            else:
                for rel, _src, w in self.inc.get(node, []):
                    if rel != "r_isa" and w >= min_weight:
                        counts[rel] += 1
        return counts

    def sem_types(self, term: str, include_refinements: bool = True) -> list[str]:
        """Types sémantiques standards (_INFO-SEM) associés au terme."""
        seen: set[str] = set()
        result: list[str] = []
        for node in self._expand_senses(term, include_refinements):
            for st in self.sem.get(node, []):
                if st not in seen:
                    seen.add(st)
                    result.append(st)
        return result