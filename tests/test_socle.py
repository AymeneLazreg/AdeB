import math
from adeb.evaluate import stratified_split
from adeb.kb import KnowledgeBase
from adeb.learn import Rule, merge_rules
from adeb.preprocess import ParsedPhrase, preprocess_phrase
from adeb.similarity import cosine


def test_cosine_known_vectors():
    u = {"a": 3.0, "b": 4.0}
    v = {"a": 6.0, "b": 8.0}
    w = {"c": 1.0}
    assert math.isclose(cosine(u, v), 1.0, rel_tol=1e-9)
    assert math.isclose(cosine(u, w), 0.0, abs_tol=1e-9)
    assert math.isclose(cosine({"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 1.0}), 0.8, rel_tol=1e-9)
    assert math.isclose(cosine({"a": 5.0, "b": 10.0}, {"a": 2.0, "b": 1.0}), 0.8, rel_tol=1e-9)


def test_fusion_commutativity_and_cosine_idempotence():
    r1 = Rule(sl={"H:vélo": 1.0, "H:objet": 0.5}, sr={"H:personne": 1.0}, rt="Po", weight=0)
    r2 = Rule(sl={"H:fusil": 1.0, "H:objet": 0.5}, sr={"H:soldat": 1.0}, rt="Po", weight=0)

    f12 = merge_rules(r1, r2)
    f21 = merge_rules(r2, r1)
    assert f12.sl == f21.sl and f12.sr == f21.sr and f12.weight == f21.weight == 1

    f11 = merge_rules(r1, r1)
    assert math.isclose(cosine(f11.sl, r1.sl), 1.0, rel_tol=1e-9)
    assert math.isclose(cosine(f11.sr, r1.sr), 1.0, rel_tol=1e-9)


def test_segmentation_10_tricky_cases():
    kb = KnowledgeBase(
        relations=[
            ("lunettes de soleil", "r_isa", "accessoire", 50),
            ("marque", "r_isa", "concept", 50),
            ("restaurant", "r_isa", "lieu", 50),
            ("fruits de mer", "r_isa", "aliment", 50),
            ("chef de gare", "r_isa", "métier", 50),
            ("Lyon", "r_isa", "ville", 50),
        ],
        sem_types={},
    )

    cases = [
        ("désert d'Algérie", "désert", "Algérie", False, True),
        ("photo d'une famille", "photo", "famille", True, False),
        ("photo de famille", "photo", "famille", False, False),
        ("fusil du soldat", "fusil", "soldat", True, True),
        ("dégâts de la tempête", "dégâts", "tempête", True, True),
        ("travail de l'ouvrier", "travail", "ouvrier", True, True),
        ("portrait de sa mère", "portrait", "mère", True, True),
        ("lunettes de soleil de marque", "lunettes de soleil", "marque", False, False),
        ("restaurant de fruits de mer", "restaurant", "fruits de mer", False, False),
        ("chef de gare de Lyon", "chef de gare", "Lyon", False, True),
    ]

    for phrase, exp_a, exp_b, exp_det, exp_def in cases:
        res = preprocess_phrase(phrase, kb=kb, lemmatize=False)
        assert res is not None, f"Échec sur {phrase}"
        assert (res.a, res.b) == (exp_a, exp_b), f"Mauvais split sur '{phrase}': {res}"
        assert res.feats == {"Det": exp_det, "Def": exp_def}, f"Mauvais DEF sur '{phrase}': {res.feats}"


def test_no_train_test_leakage():
    items = [
        (ParsedPhrase(f"a{i} de b{i}", f"a{i}", f"b{i}", "de", {"Det": False, "Def": False}), f"C{i % 3}")
        for i in range(30)
    ]
    train, test = stratified_split(items, train_ratio=0.6, seed=7)
    assert {p.raw for p, _ in train}.isdisjoint({p.raw for p, _ in test})
    assert len(train) + len(test) == len(items)
