"""Property 5: Agreement is symmetric and maximal for identical labels.

**Validates: Requirements 3.3**
"""
import math

from hypothesis import given, settings, strategies as st

from label_agreement import VERDICTS, cohen_kappa, raw_agreement

verdict = st.sampled_from(VERDICTS)
# pares (original, re-rótulo) garantem vetores do mesmo tamanho
pairs = st.lists(st.tuples(verdict, verdict), max_size=60)
labels = st.lists(verdict, max_size=60)


def _same(x: float, y: float) -> bool:
    return (math.isnan(x) and math.isnan(y)) or x == y


@settings(max_examples=300, deadline=None)
@given(pairs=pairs)
def test_agreement_is_symmetric(pairs):
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    assert _same(cohen_kappa(a, b), cohen_kappa(b, a))
    assert _same(raw_agreement(a, b), raw_agreement(b, a))


@settings(max_examples=300, deadline=None)
@given(a=labels)
def test_identical_labels_give_maximal_agreement(a):
    k = cohen_kappa(a, list(a))
    if len(set(a)) >= 2:
        assert math.isclose(k, 1.0)
    else:
        # vazio ou uma única classe: pe = 1 (ou n = 0), kappa indefinido
        assert math.isnan(k)
    if a:
        assert raw_agreement(a, list(a)) == 1.0
    else:
        assert math.isnan(raw_agreement(a, a))
