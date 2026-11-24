# Projet HAI922 — Identification de relations sémantiques « A de B » (GRASP-it)

Implémentation symbolique explicable de **GRASP-it** (Guenoune & Lafourcade, CogALex 2024) sur un sous-graphe extrait de *JeuxDeMots*, complétée par des variantes d'agrégation (`C2`), de fusion (`A3`), de comparaison discriminante/neuronale (`A7`/MLP) et d'analyse d'erreurs.

## Prérequis
- Python 3.10+ (testé sous Python 3.12)
- Dépendance optionnelle pour les baselines statistiques (`scripts/collect_report_data.py`) : `pip install scikit-learn`

## Reproduction complète
```bash
# 1. Tests unitaires (cosinus, fusion, segmentation, non-fuite train/test)
pytest -v

# 2. Expériences du socle (Exp1-Exp3 sur 5 graines) + démo explicable
python -m adeb.run --kb data/jdm_complet.json --corpus data/corpus_complet.tsv --seeds 5

# 3. Variantes et analyse d'erreurs complète
python scripts/exp_aggregation.py
python scripts/exp_fusion.py
python scripts/analyze_errors.py
python scripts/collect_report_data.py