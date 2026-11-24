from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adeb.preprocess import generate_split_candidates, simple_french_lemma

API_BASE = "https://jdm-api.demo.lirmm.fr/v0"

# IDs des relations sémantiques utiles pour les TRT (évite r_associated, r_context, r_wiki...)
TRT_TYPE_IDS = [
    3,    # r_domain
    9,    # r_has_part
    10,   # r_holo
    13,   # r_agent
    14,   # r_patient
    15,   # r_lieu
    16,   # r_instr
    17,   # r_carac
    23,   # r_carac-1
    24,   # r_agent-1
    25,   # r_instr-1
    26,   # r_patient-1
    28,   # r_lieu-1
    31,   # r_action_lieu
    41,   # r_has_conseq
    42,   # r_has_causatif
    49,   # r_time
    50,   # r_object>mater
    51,   # r_mater>object
    53,   # r_make
    54,   # r_product_of
    58,   # r_quantificateur
    70,   # r_processus>agent
    76,   # r_processus>patient
    80,   # r_processus>instr
    100,  # r_has_auteur
    113,  # r_has_social_tie_with
    121,  # r_own
    122,  # r_own-1
    136,  # r_has_auteur-1
    137,  # r_processus>agent-1
    138,  # r_processus>patient-1
    139,  # r_processus>instr-1
    142,  # r_has_topic
    153,  # r_has_prop
    171,  # r_lieu>origine
    172,  # r_depict
    173,  # r_has_prop-1
    174,  # r_quantificateur-1
]


def api_get(path: str, retries: int = 3, timeout: int = 20) -> dict | list | None:
    """Effectue un GET JSON sur l'API JDM avec gestion des 404 et réessais."""
    url = f"{API_BASE}{path}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TP-AdeB-Student/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == retries - 1:
                print(f"[WARN] HTTP {e.code} sur {url}")
                return None
            time.sleep(0.5 * (attempt + 1))
        except Exception as e:
            if attempt == retries - 1:
                print(f"[WARN] Erreur sur {url} : {e}")
                return None
            time.sleep(0.5 * (attempt + 1))
    return None


def detect_types_query_style() -> str:
    """Détecte si l'API attend types_ids=10&types_ids=15 ou types_ids=10,15."""
    repeated = "&".join(f"types_ids={tid}" for tid in (10, 15))
    if api_get(f"/relations/to/chat?{repeated}&min_weight=1&limit=1", retries=1) is not None:
        return "repeated"
    return "comma"


def build_types_query(type_ids: list[int], style: str) -> str:
    if style == "repeated":
        return "&".join(f"types_ids={tid}" for tid in type_ids)
    return "types_ids=" + ",".join(str(tid) for tid in type_ids)


def collect_corpus_terms(corpus_paths: list[Path]) -> set[str]:
    """Collecte les formes A et B (brutes + lemmes simples) présentes dans les corpus TSV."""
    terms: set[str] = set()
    for path in corpus_paths:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                phrase = row.get("phrase") or next(iter(row.values()))
                for cand in generate_split_candidates(phrase):
                    for raw_term in (cand.a, cand.b):
                        if not raw_term:
                            continue
                        terms.add(raw_term)
                        lemma = simple_french_lemma(raw_term)
                        if lemma != raw_term:
                            terms.add(lemma)
                        if raw_term[0].isupper():
                            terms.add(raw_term.lower())
    return terms


def fetch_term_info(
    term: str,
    rt_id_to_name: dict[int, str],
    trt_query: str,
    need_full: bool,
    min_weight: int,
    max_hyper: int,
) -> tuple[str, dict]:
    """
    Interroge l'API pour un terme :
    - toujours : hypernymes r_isa (id=6)
    - si need_full=True (terme du corpus) : SST (_INFO-SEM, id=36) + TRT entrantes
    """
    q = urllib.parse.quote(term, safe="")
    node = api_get(f"/node_by_name/{q}")
    if not node:
        return term, {"exists": False, "hypernyms": [], "sem_types": [], "trt_counts": {}}

    # 1. Hypernymes (r_isa = 6)
    hypernyms: list[tuple[str, float]] = []
    data_isa = api_get(f"/relations/from/{q}?types_ids=6&min_weight={min_weight}&limit=100")
    if data_isa and "relations" in data_isa:
        id_to_name = {n["id"]: n["name"] for n in data_isa.get("nodes", [])}
        best_h: dict[str, float] = {}
        for rel in data_isa["relations"]:
            if rel.get("type") == 6 and rel.get("w", 0) >= min_weight:
                dst = id_to_name.get(rel["node2"])
                if dst and dst != term and not dst.startswith("_"):
                    w = float(rel["w"])
                    if w > best_h.get(dst, 0.0):
                        best_h[dst] = w
        hypernyms = sorted(best_h.items(), key=lambda x: x[1], reverse=True)[:max_hyper]

    sem_types: list[str] = []
    trt_counts: dict[str, int] = {}

    if need_full:
        # 2. Types sémantiques standards (r_infopot = 36)
        data_sst = api_get(f"/relations/from/{q}?types_ids=36&min_weight=1&limit=100")
        if data_sst and "nodes" in data_sst:
            sem_types = sorted(
                {n["name"] for n in data_sst["nodes"] if n["name"].startswith("_INFO-SEM")}
            )

        # 3. Relations entrantes cibles (TRT)
        data_trt = api_get(f"/relations/to/{q}?{trt_query}&min_weight=1&limit=1000")
        if data_trt and "relations" in data_trt:
            counts: Counter[str] = Counter()
            for rel in data_trt["relations"]:
                rname = rt_id_to_name.get(rel.get("type"))
                if rname and rname != "r_isa" and rel.get("w", 0) >= 1:
                    counts[rname] += 1
            trt_counts = dict(counts)

    return term, {
        "exists": True,
        "full": need_full,
        "hypernyms": hypernyms,
        "sem_types": sem_types,
        "trt_counts": trt_counts,
    }


def load_cache(cache_path: Path) -> dict[str, dict]:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return {}


def save_cache(cache_path: Path, cache: dict[str, dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def fetch_batch(
    terms: list[str],
    cache: dict[str, dict],
    cache_path: Path,
    rt_id_to_name: dict[int, str],
    trt_query: str,
    need_full: bool,
    min_weight: int,
    max_hyper: int,
    workers: int,
) -> None:
    todo = [
        t for t in terms
        if t not in cache or (need_full and cache[t].get("exists") and not cache[t].get("full"))
    ]
    if not todo:
        return

    print(f"   -> {len(todo)} termes à interroger sur l'API ({len(terms) - len(todo)} déjà en cache)...")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                fetch_term_info, t, rt_id_to_name, trt_query, need_full, min_weight, max_hyper
            ): t
            for t in todo
        }
        for fut in as_completed(futures):
            term, info = fut.result()
            cache[term] = info
            done += 1
            if done % 25 == 0 or done == len(todo):
                print(f"      ... {done}/{len(todo)} récupérés")
                save_cache(cache_path, cache)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construit un sous-graphe JDM local via l'API REST")
    parser.add_argument("--corpus", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("data/jdm_subgraph.json"))
    parser.add_argument("--cache", type=Path, default=Path("data/jdm_api_cache.json"))
    parser.add_argument("--depth", type=int, default=2, help="Profondeur de récupération des hypernymes")
    parser.add_argument("--min-weight", type=int, default=10, help="Poids min pour r_isa")
    parser.add_argument("--max-hyper", type=int, default=15, help="Max hypernymes suivis par terme")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    t0 = time.perf_counter()
    print("[1/4] Chargement des types de relations depuis l'API JDM...")
    rts = api_get("/relations_types")
    if not rts:
        raise RuntimeError("Impossible de joindre l'API JeuxDeMots (/v0/relations_types).")
    rt_id_to_name = {r["id"]: r["name"] for r in rts}

    style = detect_types_query_style()
    trt_query = build_types_query(TRT_TYPE_IDS, style)
    print(f"      {len(rt_id_to_name)} types chargés (style query: {style}).")

    cache = load_cache(args.cache)
    seed_terms = sorted(collect_corpus_terms(args.corpus))
    print(f"[2/4] {len(seed_terms)} formes candidates extraites du corpus.")

    fetch_batch(
        seed_terms, cache, args.cache, rt_id_to_name, trt_query,
        need_full=True, min_weight=args.min_weight, max_hyper=args.max_hyper, workers=args.workers,
    )

    # BFS pour récupérer les hypernymes aux profondeurs 2..depth
    print(f"[3/4] Expansion des hypernymes jusqu'à profondeur {args.depth}...")
    frontier = [t for t in seed_terms if cache.get(t, {}).get("exists")]
    visited = set(frontier)
    for lvl in range(1, args.depth):
        next_terms = sorted(
            {
                dst
                for t in frontier
                for dst, _w in cache.get(t, {}).get("hypernyms", [])
                if dst not in visited
            }
        )
        print(f"   Niveau {lvl + 1} : {len(next_terms)} nouveaux hypernymes")
        if not next_terms:
            break
        fetch_batch(
            next_terms, cache, args.cache, rt_id_to_name, trt_query,
            need_full=False, min_weight=args.min_weight, max_hyper=args.max_hyper, workers=args.workers,
        )
        visited.update(next_terms)
        frontier = [t for t in next_terms if cache.get(t, {}).get("exists")]

    # Export au format attendu par KnowledgeBase.from_json()
    print(f"[4/4] Écriture du sous-graphe dans {args.out}...")
    relations_out: list[list] = []
    sem_types_out: dict[str, list[str]] = {}
    trt_counts_out: dict[str, dict[str, int]] = {}

    for term in sorted(visited):
        info = cache.get(term)
        if not info or not info.get("exists"):
            continue
        for dst, w in info.get("hypernyms", []):
            relations_out.append([term, "r_isa", dst, w])
        if info.get("sem_types"):
            sem_types_out[term] = info["sem_types"]
        if info.get("trt_counts"):
            trt_counts_out[term] = info["trt_counts"]

    payload = {
        "meta": {
            "source": "jdm-api.demo.lirmm.fr/v0",
            "depth": args.depth,
            "min_weight": args.min_weight,
            "max_hyper": args.max_hyper,
            "n_terms": len(visited),
            "n_isa_edges": len(relations_out),
        },
        "relations": relations_out,
        "sem_types": sem_types_out,
        "trt_counts": trt_counts_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    size_kb = args.out.stat().st_size / 1024
    print(f"[OK] {args.out} créé ({size_kb:.1f} Ko, {len(relations_out)} arcs r_isa) en {time.perf_counter() - t0:.1f}s.")


if __name__ == "__main__":
    main()