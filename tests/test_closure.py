import pytest

from grb_afterglow import CLOSURE_RELATIONS, evaluate_closure_relations


def test_relations_are_self_consistent():
    for rel in CLOSURE_RELATIONS:
        p = 2.3
        assert rel.p_of_beta(rel.beta_of_p(p)) == pytest.approx(p)
        assert rel.alpha_of_beta(rel.beta_of_p(p)) == pytest.approx(rel.alpha_of_p(p))


def test_ism_slow_cooling_is_identified():
    p = 2.2
    best = evaluate_closure_relations(3 * (p - 1) / 4, 0.02, (p - 1) / 2, 0.02)[0]
    assert best.relation.key == "ism_slow"
    assert best.p == pytest.approx(p)
    assert best.consistent


def test_post_jet_break_is_identified():
    best = evaluate_closure_relations(2.4, 0.03, 0.7, 0.02)[0]
    assert best.relation.key == "jet_slow"
    assert best.p == pytest.approx(2.4)


def test_results_are_sorted_by_tension():
    tests = evaluate_closure_relations(1.0, 0.05, 0.6, 0.05)
    assert [t.n_sigma for t in tests] == sorted(t.n_sigma for t in tests)
