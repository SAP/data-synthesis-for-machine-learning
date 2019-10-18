
from .testdata import adults01


def test_calculate_degree():
    from ds4ml.synthesizer import calculate_degree
    degree = calculate_degree(8140, 6, 0.05)
    assert 0 < degree <= 6 / 2


def test_greedy_bayes():
    from pandas import DataFrame
    from ds4ml.synthesizer import greedy_bayes
    network = greedy_bayes(DataFrame(adults01), epsilon=0.1)
    assert network[0][0] in adults01.columns
    assert type(network[0][1]) == list


def test_noisy_distributions():
    from ds4ml.synthesizer import noisy_distributions
    from pandas import DataFrame
    dataset = DataFrame([
        [1, 0, 40],
        [1, 1, 42],
        [0, 0, 30],
        [0, 1, 30],
        [1, 1, 36],
        [1, 1, 50],
        [0, 0, 32],
        [0, 0, 28]
    ], columns=['salary', 'sex',  'age'])
    columns = ['salary', 'sex']
    epsilon = 0.05
    noisy = noisy_distributions(dataset, columns, epsilon)
    assert noisy.shape == (4, 3)
    assert len(noisy[noisy['freq'] >= 0]) == 4
