
from pandas import DataFrame
from ds4ml.evaluator import BiFrame, split_feature_class
from ds4ml.dataset import DataSet
from numpy import array_equal

from .testdata import adults01, adults02


def test_corr():
    frame = BiFrame(DataFrame(adults01), DataFrame(adults02))
    a_mi, b_mi = frame.corr()
    from numpy import allclose, alltrue
    assert allclose(a_mi, a_mi.T)
    assert alltrue(a_mi >= 0.0)
    assert alltrue(a_mi <= 1.0)
    assert allclose(b_mi, b_mi.T)
    assert alltrue(b_mi >= 0.0)
    assert alltrue(b_mi <= 1.0)


def test_dist():
    frame = BiFrame(DataFrame(adults01), DataFrame(adults02))
    columns = ['age', 'education', 'relationship', 'salary', 'birth']
    for col in columns:
        bins, counts = frame.dist(col)
        assert len(bins) == len(counts[0])
        assert len(bins) == len(counts[1])


def test_describe():
    frame = BiFrame(DataFrame(adults01), DataFrame(adults02))
    desc = frame.describe()
    from numpy import alltrue
    columns = ['age', 'birth', 'education', 'relationship', 'salary']
    # 'sex' not in adults01
    assert array_equal(desc.columns, columns)
    assert alltrue(desc >= 0.0)
    assert alltrue(desc <= 1.0)
    assert array_equal(desc.index, ['err', 'jsd'])


def test_split_feature_class():
    frame = DataSet(adults01[['age', 'relationship', 'salary']].head(10)).encode()
    features1, class1 = split_feature_class('birth', frame)
    assert features1.equals(frame)
    assert class1 is None

    features2, class2 = split_feature_class('age', frame)
    assert features2.equals(frame)
    assert class2 is None

    features3, class3 = split_feature_class('salary', frame)
    assert len(features3.columns) == 4
    assert class3.name == 'salary_>50K'

    features4, class4 = split_feature_class('relationship', frame)
    assert len(features4.columns) == 3
    assert class4.min() == 0
    assert class4.max() == 2


def test_classify_one_class():
    frame = BiFrame(DataFrame(adults01), DataFrame(adults02))
    matrix = frame.classify('salary')
    assert len(matrix) == 2
    assert array_equal(matrix[0].columns, ['<=50K', '>50K'])
    assert array_equal(matrix[0].index, ['<=50K', '>50K'])


def test_classify_multiple_classes():
    frame = BiFrame(DataFrame(adults01), DataFrame(adults02))
    matrix = frame.classify('education')
    assert len(matrix) == 2
    columns = ['11th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors',
               'Doctorate', 'HS-grad', 'Masters', 'Some-college']
    assert array_equal(matrix[0].columns, columns)
    assert array_equal(matrix[0].index, columns)
    assert array_equal(matrix[1].columns, columns)
    assert array_equal(matrix[1].index, columns)


def test_to_html():
    frame = BiFrame(DataFrame(adults01), DataFrame(adults02))
    report = 'a.html'
    frame.to_html(report, labels=['education'])
    import os.path
    assert os.path.isfile(report)
    if os.path.exists(report):
        os.remove(report)
