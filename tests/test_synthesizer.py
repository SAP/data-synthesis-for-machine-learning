
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
    ], columns=['salary', 'sex', 'age'])
    columns = ['salary', 'sex']
    epsilon = 0.05
    noisy = noisy_distributions(dataset, columns, epsilon)
    assert noisy.shape == (4, 3)
    assert len(noisy[noisy['freq'] >= 0]) == 4


def test_noisy_distributions_for_one_column():
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
    ], columns=['salary', 'sex', 'age'])
    columns = ['sex']
    epsilon = 0.05
    noisy = noisy_distributions(dataset, columns, epsilon)
    assert noisy.shape == (2, 2)
    assert len(noisy[noisy['freq'] >= 0]) == 2

    columns = ['age']
    epsilon = 0.05
    noisy = noisy_distributions(dataset[['age']], columns, epsilon)
    assert noisy.shape[1] == 2


def test_noisy_conditionals():
    from ds4ml.synthesizer import noisy_conditionals
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
    ], columns=['salary', 'sex', 'age'])
    epsilon = 0.05
    network = [('salary', ['sex']), ('age', ['sex', 'salary'])]
    noisy = noisy_conditionals(network, dataset, epsilon)
    assert len(noisy['sex']) == 2
    assert (1.0 - sum(noisy['sex'])) < 1e-6
    assert len(noisy['salary']) == 2
    assert '[0]' in noisy['salary']
    assert '[1]' in noisy['salary']
    assert len(noisy['age']) == 4
    assert '[0, 0]' in noisy['age']
    assert '[0, 1]' in noisy['age']
    assert '[1, 0]' in noisy['age']
    assert '[1, 1]' in noisy['age']
