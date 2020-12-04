
from ds4ml.metrics import (pairwise_mutual_information,
                           relative_error, error_rate)


def test_pairwise_mutual_information():
    from pandas import DataFrame
    from .testdata import adults01
    frame = DataFrame(adults01)
    mi = pairwise_mutual_information(frame)
    from numpy import allclose, alltrue
    assert allclose(mi, mi.T)
    assert alltrue(mi >= 0.0)
    assert alltrue(mi <= 1.0)


def test_error_rate():
    from sklearn.metrics import accuracy_score
    from numpy.random import randint
    from numpy import isclose
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    # its confusion matrix:
    # 2 0 0
    # 0 0 1
    # 1 0 2
    assert isclose(error_rate(y_true, y_pred), 0.3333)
    for _ in range(10):
        y_true = randint(0, 20, 100)
        y_pred = randint(0, 20, 100)
        assert isclose(error_rate(y_true, y_pred),
                       1 - accuracy_score(y_true, y_pred))


def test_relative_error():
    from numpy import array, random
    a = array([0, 1, 0])
    b = array([1, 1, 0])
    assert relative_error(a, b) == 0.3333

    for _ in range(10):
        size = random.randint(5, 100)
        x = random.choice(20, size)
        y = random.choice(20, size)
        assert 0.0 <= relative_error(x, y) <= 1.0
        x = random.rand(size)
        y = random.rand(size)
        assert 0.0 <= relative_error(x, y) <= 1.0

