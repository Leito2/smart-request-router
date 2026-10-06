"""Judge validation: agreement between a judge and human labels."""
from collections import Counter
from collections.abc import Hashable, Sequence


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Cohen's kappa for two raters over the same items (1 = perfect, 0 = chance level)."""
    if len(a) != len(b) or not a:
        raise ValueError("raters must label the same non-empty set of items")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / (n * n)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)
