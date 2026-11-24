from __future__ import annotations

import math
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adeb.classify import classify
from adeb.evaluate import bootstrap_f1_ci, compute_metrics, evaluate_once, stratified_split
from adeb.kb import KnowledgeBase
from adeb.learn import learn_greedy
from adeb.preprocess import ParsedPhrase, preprocess_phrase
from adeb.run import load_dataset
from adeb.signature import Config, Vec, build_signature


def mcnemar_test(y_true: list[str], pred_a: list[str], pred_b: list[str]) -> tuple[int, int, float, float]:
    """Test de McNemar (avec correction de continuité) entre deux modèles sur le même jeu de test."""
    b = sum(1 for g, a, pb in zip(y_true, pred_a, pred_b) if a == g and pb != g)
    c = sum(1 for g, a, pb in zip(y_true, pred_a, pred_b) if a != g and pb == g)
    if b + c == 0:
        return b, c, 0.0, 1.0
    chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
    # p-value exacte approx pour chi2 à 1 ddl : erfc(sqrt(chi2 / 2))
    pval = math.erfc(math.sqrt(chi2 / 2.0))
    return b, c, chi2, pval


def kfold_stratified(
    items: list[tuple[ParsedPhrase, str]], k: int = 5, seed: int = 42
) -> list[tuple[list[tuple[ParsedPhrase, str]], list[tuple[ParsedPhrase, str]]]]:
    rng = random.Random(seed)
    by_label: dict[str, list[tuple[ParsedPhrase, str]]] = defaultdict(list)
    for it in items:
        by_label[it[1]].append(it)
    folds: list[list[tuple[ParsedPhrase, str]]] = [[] for _ in range(k)]
    for _lbl, grp in sorted(by_label.items()):
        shuf = list(grp)
        rng.shuffle(shuf)
        for i, it in enumerate(shuf):
            folds[i % k].append(it)
    splits = []
    for i in range(k):
        val = list(folds[i])
        tr = [it for j, f in enumerate(folds) if j != i for it in f]
        rng.shuffle(tr)
        splits.append((tr, val))
    return splits


def build_train_vecs(
    kb: KnowledgeBase, items: list[tuple[ParsedPhrase, str]], cfg: Config
) -> list[tuple[Vec, Vec, str]]:
    return [
        (
            build_signature(kb, p.a, cfg, p.feats, "A"),
            build_signature(kb, p.b, cfg, p.feats, "B"),
            rt,
        )
        for p, rt in items
    ]


def main() -> None:
    print("=== 0. PACKAGES ML DISPONIBLES ===")
    avail: dict[str, bool] = {}
    for pkg in ("numpy", "scipy", "sklearn", "torch", "transformers", "sentence_transformers"):
        try:
            mod = __import__(pkg)
            avail[pkg] = True
            print(f"  {pkg}: OK ({getattr(mod, '__version__', '?')})")
        except Exception:
            avail[pkg] = False
            print(f"  {pkg}: NON")

    print("\n=== 1. CHARGEMENT KB & DONNEES ===")
    t0 = time.perf_counter()
    kb = KnowledgeBase.from_json(Path("data/jdm_complet.json"))
    kb_load_s = time.perf_counter() - t0
    items = load_dataset(Path("data/corpus_complet.tsv"), kb)
    print(f"Temps chargement KB : {kb_load_s:.3f} s | Exemples exploitables : {len(items)}")

    train_items, test_items = stratified_split(items, train_ratio=50 / 80, seed=0)
    print(f"Split seed=0 : {len(train_items)} train / {len(test_items)} test")

    # ------------------------------------------------------------------
    # 2. COMPROMIS QUALITE / TEMPS / TAILLE (No Trim vs Trim) + DEF PAR CLASSE
    # ------------------------------------------------------------------
    print("\n=== 2. COMPROMIS QUALITE / TEMPS / TAILLE (seed=0) ===")
    cfg_nodef = Config(use_def=False, trim=True, fusion_threshold=0.50, classify_agg="mean")
    cfg_notrim = Config(use_def=True, trim=False, fusion_threshold=0.50, classify_agg="mean")
    cfg_trim = Config(use_def=True, trim=True, fusion_threshold=0.50, classify_agg="mean")

    for name, cfg in [("No Trim (+DEF)", cfg_notrim), ("Trim (+DEF)", cfg_trim)]:
        t_sig0 = time.perf_counter()
        tr_vecs = build_train_vecs(kb, train_items, cfg)
        sig_s = time.perf_counter() - t_sig0

        t_learn0 = time.perf_counter()
        rules = learn_greedy(tr_vecs, cfg)
        learn_s = time.perf_counter() - t_learn0

        size_kb = len(pickle.dumps(rules)) / 1024.0
        init_lens_a = [len(sa) for sa, _, _ in tr_vecs]
        init_lens_b = [len(sb) for _, sb, _ in tr_vecs]
        fused_rules = [r for r in rules if r.weight > 0]
        singletons = [r for r in rules if r.weight == 0]
        rule_lens_a = [len(r.sl) for r in rules]
        rule_lens_b = [len(r.sr) for r in rules]
        fused_lens_a = [len(r.sl) for r in fused_rules] or [0]
        fused_lens_b = [len(r.sr) for r in fused_rules] or [0]

        metrics, yt, yp = evaluate_once(kb, train_items, test_items, cfg)
        lo, hi = bootstrap_f1_ci(yt, yp, n_resamples=1000, seed=42)
        ms_per_ex = (metrics.classify_seconds * 1000.0) / len(test_items)
        print(
            f"[{name}] F1={metrics.macro_f1:.3f} (IC95 [{lo:.3f}, {hi:.3f}]) | "
            f"Règles={len(rules)} (fusionnées={len(fused_rules)}, singletons={len(singletons)}) | "
            f"Taille={size_kb:.1f} Ko ({size_kb/1024:.2f} Mo)"
        )
        print(
            f"  Temps : sig_train={sig_s*1000:.1f} ms | learn={learn_s*1000:.1f} ms | "
            f"test_total={metrics.classify_seconds*1000:.1f} ms | {ms_per_ex:.2f} ms/ex"
        )
        print(
            f"  Taille moy. signatures : avant fusion A={sum(init_lens_a)/len(init_lens_a):.1f}, "
            f"B={sum(init_lens_b)/len(init_lens_b):.1f} | "
            f"règles finales A={sum(rule_lens_a)/len(rule_lens_a):.1f}, B={sum(rule_lens_b)/len(rule_lens_b):.1f} | "
            f"règles fusionnées (w>0) A={sum(fused_lens_a)/len(fused_lens_a):.1f}, B={sum(fused_lens_b)/len(fused_lens_b):.1f}"
        )

    print("\n=== 3. EFFET DE DEF PAR CLASSE (H+TRT+SST vs +DEF, Trim=True, seed=0) ===")
    m_nodef, yt_nd, yp_nd = evaluate_once(kb, train_items, test_items, cfg_nodef)
    m_def, yt_d, yp_d = evaluate_once(kb, train_items, test_items, cfg_trim)
    b, c, chi2, pval = mcnemar_test(yt_d, yp_d, yp_nd)
    print(f"F1 sans DEF={m_nodef.macro_f1:.3f} -> avec DEF={m_def.macro_f1:.3f} | McNemar (+DEF vs -DEF): b={b}, c={c}, chi2={chi2:.2f}, p={pval:.4f}")
    print(f"{'Classe':6s} | {'F1 sans DEF':11s} | {'F1 avec DEF':11s} | {'Delta':7s}")
    print("-" * 45)
    deltas = []
    for cl in sorted(m_def.per_class.keys()):
        f0 = m_nodef.per_class[cl]["f1"]
        f1 = m_def.per_class[cl]["f1"]
        deltas.append((f1 - f0, cl, f0, f1))
    for d, cl, f0, f1 in sorted(deltas):
        print(f"{cl:6s} | {f0:11.3f} | {f1:11.3f} | {d:+7.3f}")

    # ------------------------------------------------------------------
    # 4. SENSIBILITE A L'ORDRE DES EXEMPLES (split fixe seed=0, 15 permutations)
    # ------------------------------------------------------------------
    print("\n=== 4. SENSIBILITE A L'ORDRE D'ENTRAINEMENT (15 permutations, split seed=0 fixe) ===")
    ord_f1s, ord_rules = [], []
    for perm_seed in range(15):
        shuf_train = list(train_items)
        random.Random(100 + perm_seed).shuffle(shuf_train)
        m_perm, _, _ = evaluate_once(kb, shuf_train, test_items, cfg_trim)
        ord_f1s.append(m_perm.macro_f1)
        ord_rules.append(m_perm.n_rules)
    f1_mean = sum(ord_f1s) / len(ord_f1s)
    f1_std = math.sqrt(sum((x - f1_mean) ** 2 for x in ord_f1s) / len(ord_f1s))
    r_mean = sum(ord_rules) / len(ord_rules)
    r_std = math.sqrt(sum((x - r_mean) ** 2 for x in ord_rules) / len(ord_rules))
    print(
        f"F1 macro : {f1_mean:.3f} ± {f1_std:.3f} (min={min(ord_f1s):.3f}, max={max(ord_f1s):.3f}) | "
        f"Règles : {r_mean:.1f} ± {r_std:.1f} (min={min(ord_rules)}, max={max(ord_rules)})"
    )

    # ------------------------------------------------------------------
    # 5. VALIDATION CROISEE 5-FOLD SUR LE TRAIN POUR THRESH (sans fuite sur le test)
    # ------------------------------------------------------------------
    print("\n=== 5. VALIDATION CROISEE 5-FOLD SUR LE TRAIN (Selection de thresh sans fuite) ===")
    folds = kfold_stratified(train_items, k=5, seed=42)
    best_th, best_cv_f1 = 0.50, -1.0
    for th in (0.35, 0.45, 0.50, 0.55, 0.65):
        cfg_cv = Config(use_def=True, trim=True, fusion_threshold=th, fusion_agg="mean")
        cv_f1s, cv_rules = [], []
        for f_tr, f_val in folds:
            m_cv, _, _ = evaluate_once(kb, f_tr, f_val, cfg_cv)
            cv_f1s.append(m_cv.macro_f1)
            cv_rules.append(m_cv.n_rules)
        m_f1 = sum(cv_f1s) / len(cv_f1s)
        s_f1 = math.sqrt(sum((x - m_f1) ** 2 for x in cv_f1s) / len(cv_f1s))
        print(f"  CV thresh={th:.2f} : F1={m_f1:.3f} ± {s_f1:.3f} | Règles moy.={sum(cv_rules)/len(cv_rules):.1f}")
        if m_f1 > best_cv_f1:
            best_cv_f1 = m_f1
            best_th = th
    print(f"Meilleur seuil sélectionné sur le train seul : thresh={best_th:.2f} (CV F1={best_cv_f1:.3f})")

    # ------------------------------------------------------------------
    # 6. IC 95% BOOTSTRAP + TEST DE MCNEMAR SUR LES VARIANTES (seed=0)
    # ------------------------------------------------------------------
    print("\n=== 6. VARIANTES : IC 95% BOOTSTRAP & TEST DE MCNEMAR VS SOCLE (seed=0) ===")
    variants = [
        ("Socle (mean, th=0.50)", cfg_trim),
        ("Agg min", Config(use_def=True, trim=True, fusion_threshold=0.50, classify_agg="min")),
        ("Agg product", Config(use_def=True, trim=True, fusion_threshold=0.50, classify_agg="product")),
        ("Fusion th=0.35 (mean)", Config(use_def=True, trim=True, fusion_threshold=0.35, fusion_agg="mean")),
        ("Fusion th=0.65 (mean)", Config(use_def=True, trim=True, fusion_threshold=0.65, fusion_agg="mean")),
        ("Fusion th=0.50 (both)", Config(use_def=True, trim=True, fusion_threshold=0.50, fusion_agg="both")),
    ]
    base_metrics, y_true_base, y_pred_base = evaluate_once(kb, train_items, test_items, cfg_trim)
    for label, vcfg in variants:
        vm, yt, yp = evaluate_once(kb, train_items, test_items, vcfg)
        lo, hi = bootstrap_f1_ci(yt, yp, n_resamples=1000, seed=42)
        b, c, chi2, pval = mcnemar_test(yt, y_pred_base, yp)
        print(
            f"  {label:24s} | F1={vm.macro_f1:.3f} [{lo:.3f}, {hi:.3f}] | "
            f"Règles={vm.n_rules:3d} | McNemar(socle>var={b}, var>socle={c}, p={pval:.4f})"
        )

    # ------------------------------------------------------------------
    # 7. MATRICE DE CONFUSION 15x15 + ANALYSE QUANTITATIVE DES ERREURS + MULTI-LABEL / REJET
    # ------------------------------------------------------------------
    print("\n=== 7. MATRICE DE CONFUSION 15x15 (seed=0) ===")
    labels = sorted(set(y_true_base))
    header = "Or\\Pr  " + " ".join(f"{l:>3s}" for l in labels)
    print(header)
    for g in labels:
        row = " ".join(f"{base_metrics.confusion.get((g, p), 0):3d}" for p in labels)
        print(f"{g:6s} {row}")

    print("\n=== 8. ANALYSE QUANTITATIVE DES ERREURS, TOP-K ET REJET (seed=0) ===")
    tr_vecs_base = build_train_vecs(kb, train_items, cfg_trim)
    rules_base = learn_greedy(tr_vecs_base, cfg_trim)

    top1_ok = 0
    top2_ok = 0
    top3_ok = 0
    margins_all: list[tuple[float, bool]] = []
    err_records = []
    ambiguous_pairs = {
        ("AC", "Po"), ("Po", "AC"), ("A", "P"), ("P", "A"),
        ("D", "Po"), ("Po", "D"), ("Q", "M"), ("M", "Q"),
        ("Co", "Ca"), ("Ca", "Co"), ("D", "AC"), ("AC", "D"),
    }

    for p, gold in test_items:
        sa = build_signature(kb, p.a, cfg_trim, p.feats, "A")
        sb = build_signature(kb, p.b, cfg_trim, p.feats, "B")
        ranking, exp = classify(rules_base, sa, sb, cfg_trim, explain=True)
        preds = [rt for _, rt in ranking]
        s1 = ranking[0][0]
        s2 = ranking[1][0] if len(ranking) > 1 else 0.0
        margin = s1 - s2
        ok = preds[0] == gold
        margins_all.append((margin, ok))
        if ok:
            top1_ok += 1
        if gold in preds[:2]:
            top2_ok += 1
        if gold in preds[:3]:
            top3_ok += 1
        if not ok:
            kb_poor = (not kb.exists(p.a)) or (not kb.exists(p.b)) or (len(sa) <= 2) or (len(sb) <= 3)
            singleton_asym = (exp is not None and exp.rule_weight == 0 and abs(exp.sim_a - exp.sim_b) >= 0.25)
            close_top2 = (gold == preds[1] and margin <= 0.03)
            sem_border = (gold, preds[0]) in ambiguous_pairs
            err_records.append(
                {
                    "raw": p.raw, "gold": gold, "pred": preds[0], "rank2": preds[1],
                    "margin": margin, "kb_poor": kb_poor, "singleton_asym": singleton_asym,
                    "close_top2": close_top2, "sem_border": sem_border,
                    "w": exp.rule_weight if exp else -1,
                    "len_sa": len(sa), "len_sb": len(sb),
                }
            )

    n_test = len(test_items)
    n_err = len(err_records)
    print(
        f"Exactitude Top-1 : {top1_ok}/{n_test} ({top1_ok/n_test:.3f}) | "
        f"Top-2 : {top2_ok}/{n_test} ({top2_ok/n_test:.3f}) | "
        f"Top-3 : {top3_ok}/{n_test} ({top3_ok/n_test:.3f})"
    )
    in_top2_among_err = sum(1 for e in err_records if e["gold"] == e["rank2"])
    print(f"Total erreurs : {n_err} | Classe de référence au rang 2 : {in_top2_among_err}/{n_err} ({100*in_top2_among_err/n_err:.1f}%)")
    print("Catégorisation des erreurs (non exclusif) :")
    print(f"  - Couverture KB faible (terme absent ou signature pauvre) : {sum(e['kb_poor'] for e in err_records)} ({100*sum(e['kb_poor'] for e in err_records)/n_err:.1f}%)")
    print(f"  - Règle singleton (w=0) déclenchée                         : {sum(e['w']==0 for e in err_records)} ({100*sum(e['w']==0 for e in err_records)/n_err:.1f}%)")
    print(f"  - Règle singleton asymétrique (w=0 & |simA-simB|>=0.25)    : {sum(e['singleton_asym'] for e in err_records)} ({100*sum(e['singleton_asym'] for e in err_records)/n_err:.1f}%)")
    print(f"  - Frontière sémantique/ambiguïté connue (AC/Po, A/P, Q/M..): {sum(e['sem_border'] for e in err_records)} ({100*sum(e['sem_border'] for e in err_records)/n_err:.1f}%)")
    print(f"  - Marge très faible (<=0.03) avec or au rang 2             : {sum(e['close_top2'] for e in err_records)} ({100*sum(e['close_top2'] for e in err_records)/n_err:.1f}%)")

    # Partition exclusive prioritaire pour un tableau synthétique à 100%
    cat_excl = Counter()
    for e in err_records:
        if e["kb_poor"]:
            cat_excl["1. Lacune / couverture KB faible"] += 1
        elif e["singleton_asym"]:
            cat_excl["2. Règle singleton (w=0) asymétrique"] += 1
        elif e["sem_border"] or e["close_top2"]:
            cat_excl["3. Ambiguïté sémantique / frontière proche"] += 1
        else:
            cat_excl["4. Polysémie ou sur-généralisation de règle"] += 1
    print("Partition exclusive des erreurs (somme = 100%) :")
    for k, v in sorted(cat_excl.items()):
        print(f"  {k:45s} : {v:2d} ({100*v/n_err:.1f}%)")

    print("\nCourbe couverture / précision selon le seuil de rejet sur la marge (s1 - s2) :")
    for delta in (0.00, 0.01, 0.02, 0.03, 0.05, 0.08):
        kept = [ok for m, ok in margins_all if m >= delta]
        cov = len(kept) / len(margins_all)
        acc_kept = sum(kept) / len(kept) if kept else 0.0
        print(f"  Seuil marge >= {delta:.2f} : couverture={100*cov:5.1f}% ({len(kept)}/{len(margins_all)}) | précision acceptée={acc_kept:.3f}")

    # ------------------------------------------------------------------
    # 9. COURBE D'APPRENTISSAGE (E5 : 10, 20, 35, 50 exemples par classe)
    # ------------------------------------------------------------------
    print("\n=== 9. COURBE D'APPRENTISSAGE (seed=0, test fixe=453) ===")
    by_lbl_tr: dict[str, list[tuple[ParsedPhrase, str]]] = defaultdict(list)
    for it in train_items:
        by_lbl_tr[it[1]].append(it)
    for n_per_class in (5, 10, 20, 35, 50):
        sub_tr = []
        for _l, grp in sorted(by_lbl_tr.items()):
            sub_tr.extend(grp[: min(n_per_class, len(grp))])
        random.Random(0).shuffle(sub_tr)
        m_sub, _, _ = evaluate_once(kb, sub_tr, test_items, cfg_trim)
        print(f"  n={n_per_class:2d}/classe (total train={len(sub_tr):3d}) -> F1={m_sub.macro_f1:.3f} | Règles={m_sub.n_rules:3d}")

    # ------------------------------------------------------------------
    # 10. BASELINES STATISTIQUES / DISCRIMINANTES (A7 + TF-IDF texte brut)
    # ------------------------------------------------------------------
    print("\n=== 10. COMPARAISON BASELINES DISCRIMINANTES / NEURONALES (seed=0) ===")
    if avail.get("sklearn"):
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier

        # A7 : Régression logistique sur les mêmes signatures [s(A); s(B)]
        def to_feat_dict(p: ParsedPhrase) -> dict[str, float]:
            sa = build_signature(kb, p.a, cfg_trim, p.feats, "A")
            sb = build_signature(kb, p.b, cfg_trim, p.feats, "B")
            d = {f"A|{k}": v for k, v in sa.items()}
            d.update({f"B|{k}": v for k, v in sb.items()})
            return d

        dv = DictVectorizer(sparse=True)
        X_tr_sig = dv.fit_transform([to_feat_dict(p) for p, _ in train_items])
        y_tr = [rt for _, rt in train_items]
        X_te_sig = dv.transform([to_feat_dict(p) for p, _ in test_items])
        y_te = [rt for _, rt in test_items]

        t0 = time.perf_counter()
        clf_lr = LogisticRegression(max_iter=1000, C=2.0)
        clf_lr.fit(X_tr_sig, y_tr)
        lr_train_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        yp_lr = list(clf_lr.predict(X_te_sig))
        lr_test_ms = (time.perf_counter() - t0) * 1000.0
        m_lr = compute_metrics(y_te, yp_lr)
        lo_lr, hi_lr = bootstrap_f1_ci(y_te, yp_lr, 1000, seed=42)
        b, c, _, pval = mcnemar_test(y_te, y_pred_base, yp_lr)
        print(
            f"  [A7] LogReg sur signatures [s(A);s(B)] : F1={m_lr.macro_f1:.3f} [{lo_lr:.3f}, {hi_lr:.3f}] | "
            f"Acc={m_lr.accuracy:.3f} | train={lr_train_ms:.1f} ms, test={lr_test_ms:.1f} ms | "
            f"McNemar(GRASP>LR={b}, LR>GRASP={c}, p={pval:.4f})"
        )

        # MLP léger sur les signatures (comparaison réseau de neurones sur mêmes traits)
        t0 = time.perf_counter()
        clf_mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=0)
        clf_mlp.fit(X_tr_sig, y_tr)
        mlp_train_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        yp_mlp = list(clf_mlp.predict(X_te_sig))
        mlp_test_ms = (time.perf_counter() - t0) * 1000.0
        m_mlp = compute_metrics(y_te, yp_mlp)
        lo_mlp, hi_mlp = bootstrap_f1_ci(y_te, yp_mlp, 1000, seed=42)
        b, c, _, pval = mcnemar_test(y_te, y_pred_base, yp_mlp)
        print(
            f"  [MLP] Réseau neuronal sur signatures   : F1={m_mlp.macro_f1:.3f} [{lo_mlp:.3f}, {hi_mlp:.3f}] | "
            f"Acc={m_mlp.accuracy:.3f} | train={mlp_train_ms:.1f} ms, test={mlp_test_ms:.1f} ms | "
            f"McNemar(GRASP>MLP={b}, MLP>GRASP={c}, p={pval:.4f})"
        )

        # Baseline texte brut (sans KB) : TF-IDF mots + n-grammes de caractères
        vec_txt = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
        X_tr_txt = vec_txt.fit_transform([p.raw for p, _ in train_items])
        X_te_txt = vec_txt.transform([p.raw for p, _ in test_items])
        t0 = time.perf_counter()
        clf_txt = LogisticRegression(max_iter=1000, C=3.0)
        clf_txt.fit(X_tr_txt, y_tr)
        txt_train_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        yp_txt = list(clf_txt.predict(X_te_txt))
        txt_test_ms = (time.perf_counter() - t0) * 1000.0
        m_txt = compute_metrics(y_te, yp_txt)
        lo_txt, hi_txt = bootstrap_f1_ci(y_te, yp_txt, 1000, seed=42)
        print(
            f"  [Txt] TF-IDF char (sans KB) + LogReg   : F1={m_txt.macro_f1:.3f} [{lo_txt:.3f}, {hi_txt:.3f}] | "
            f"Acc={m_txt.accuracy:.3f} | train={txt_train_ms:.1f} ms, test={txt_test_ms:.1f} ms"
        )
    else:
        print("  scikit-learn non installé (installer avec: pip install scikit-learn)")

    # ------------------------------------------------------------------
    # 11. CAS EMBLÉMATIQUES DE L'ARTICLE ET DU CARNET
    # ------------------------------------------------------------------
    print("\n=== 11. PREDICTIONS SUR LES CAS EMBLEMATIQUES ===")
    demo_phrases = [
        "photo de famille",
        "photo d'une famille",
        "portrait de sa mère",
        "travail du bois",
        "travail de l'ouvrier",
        "restaurant de fruits de mer",
        "richesse du royaume",
        "avocat d'une femme battue",
    ]
    for raw in demo_phrases:
        parsed = preprocess_phrase(raw, kb)
        if not parsed:
            print(f"  '{raw}' -> ECHEC segmentation")
            continue
        sa = build_signature(kb, parsed.a, cfg_trim, parsed.feats, "A")
        sb = build_signature(kb, parsed.b, cfg_trim, parsed.feats, "B")
        rank, exp = classify(rules_base, sa, sb, cfg_trim, explain=True)
        top3 = [(rt, round(sc, 3)) for sc, rt in rank[:3]]
        print(f"  '{raw}' -> A='{parsed.a}', B='{parsed.b}', {parsed.feats} | top3={top3}")


if __name__ == "__main__":
    main()