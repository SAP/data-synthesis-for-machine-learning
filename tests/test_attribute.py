

from math import isclose

from numpy import random, array_equal
from pandas import Series
from ds4ml.attribute import Attribute
from ds4ml.utils import randomize_string

size = 30


def test_integer_attribute():
    ints = random.randint(1, 100, size)
    attr = Attribute(Series(ints), name='ID', categorical=False)
    assert attr.atype == 'integer'
    assert attr.name == 'ID'
    assert attr._min >= 1
    assert attr._max <= 100
    assert len(attr.bins) == 20
    assert isclose(sum(attr.prs), 1.0)

    from .testdata import adults01
    attr = Attribute(adults01['age'])
    assert attr.atype == 'integer'


def test_float_attribute():
    floats = random.uniform(1, 100, size)
    attr = Attribute(Series(floats, name='Float'))
    assert attr.atype == 'float'
    assert attr._min >= 1
    assert attr._max <= 100
    assert len(attr.bins) == 20
    assert isclose(sum(attr.prs), 1.0)


def test_string_attribute():
    strings = list(map(lambda x: randomize_string(5), range(size)))
    attr = Attribute(Series(strings, name='String'), categorical=True)
    assert attr.atype == 'string'
    assert attr._min == 5
    assert attr.categorical


def test_set_domain_for_integer_attribute():
    ints = random.randint(1, 100, size)
    attr = Attribute(Series(ints, name='Integer'))
    assert attr._min >= 1
    assert attr._max <= 100
    attr.domain = [-2, 120]
    assert attr._min == -2
    assert attr._max == 120


def test_set_domain_for_integer_categorical_attribute():
    ints = random.randint(1, 100, size)
    attr = Attribute(Series(ints, name='Integer'), categorical=True)
    assert attr.bins[0] >= 1
    assert attr.bins[-1] <= 100
    attr.domain = [-2, 120]
    assert attr.bins[0] == -2
    assert attr.bins[-1] == 120


def test_set_domain_for_float_attribute():
    floats = random.uniform(1, 100, size)
    attr = Attribute(Series(floats, name='Float'))
    assert attr._min >= 1
    assert attr._max <= 100
    attr.domain = [-2, 120]
    assert attr._min == -2
    assert attr._max == 120


def test_set_domain_for_string_attribute():
    strings = list(map(lambda x: randomize_string(5), range(size)))
    attr = Attribute(Series(strings, name='String'), categorical=True)
    bins = attr.bins
    attr.domain = ['a', 'b', 'China', 'USA']
    assert len(bins) + 4 == len(attr.bins)


def test_set_domain_for_datetime_attribute():
    dates = ['05/29/1988', '06/22/1988', '07/30/1992', '07/30/1992',
             '11/12/2000', '01/02/2001', '01/02/2001', '12/03/2001',
             '07/09/2002', '10/22/2002']
    attr = Attribute(Series(dates, name='String'), categorical=True)
    bins = attr.bins
    attr.domain = ['07/01/1997', '12/20/1999', '01/01/2004']
    assert len(bins) + 3 == len(attr.bins)


def test_counts_numerical_attribute():
    ints = random.randint(1, 100, size)
    attr = Attribute(Series(ints, name='Integer'))
    counts = attr.counts()
    assert sum(counts) == 30
    assert len(counts) == 20
    counts = attr.counts(bins=[0, 10, 20, 30, 100])
    assert sum(counts) == 30
    assert len(counts) == 4

    # categorical ints
    attr = Attribute(Series([1, 10, 11, 10, 20, 15, 16, 25], name='Integer'),
                     categorical=True)
    counts = attr.counts()
    assert sum(counts) == 8
    assert len(counts) == 7
    counts = attr.counts(bins=[5, 10, 15])
    assert sum(counts) == 3
    assert len(counts) == 3


def test_counts_datetimes():
    dates = ['05/29/1988', '06/22/1988', '07/30/1992', '07/30/1992',
             '11/12/2000', '01/02/2001', '01/02/2001', '12/03/2001',
             '07/09/2002', '10/22/2002']
    attr = Attribute(Series(dates, name='DateTime'), categorical=True)
    counts = attr.counts()
    assert sum(counts) == len(dates)
    assert array_equal(counts, [1, 1, 2, 1, 2, 1, 1, 1])

    counts = attr.counts(bins=['12/03/2001', '10/22/2002'])
    assert array_equal(counts, [1, 1])


def test_counts_categorical_attribute():
    ints = random.randint(1, 10, size)
    attr = Attribute(Series(ints, name='Integer'), categorical=True)
    assert sum(attr.counts()) == 30


def test_choice_integers():
    ints = random.randint(1, 100, size)
    attr = Attribute(Series(ints, name='Integer'))
    assert len(attr.bins) == 20
    choices = attr.choice()
    assert len(choices) == size


def test_choice_floats():
    floats = random.uniform(1, 10, size)
    attr = Attribute(Series(floats, name='Float'))
    assert len(attr.bins) == 20
    choices = attr.choice()
    assert len(choices) == size


def test_choice_strings():
    strings = list(map(lambda x: randomize_string(5), range(size)))
    attr = Attribute(Series(strings, name='String'))
    choices = attr.choice()
    assert len(choices) == size


def test_choice_datetimes():
    dates = ['05/29/1988', '06/22/1988', '07/30/1992', '01/02/2001',
             '11/12/2000', '07/09/2002', '08/30/1998', '06/03/1997',
             '10/22/2002', '12/03/2001']
    attr = Attribute(Series(dates, name='DateTime'))
    choices = attr.choice()
    assert len(choices) == len(dates)


def test_bin_indexes_ints():
    ints = [3, 5, 7, 8, 7, 1, 10, 30, 16, 19]
    attr = Attribute(Series(ints), name='ID', categorical=False)
    indexes = attr.bin_indexes()
    assert len(indexes) == len(ints)


def test_bin_indexes_datetimes():
    dates = ['05/29/1988', '06/22/1988', '07/30/1992', '07/30/1992',
             '11/12/2000', '01/02/2001', '01/02/2001', '12/03/2001',
             '07/09/2002', '10/22/2002']
    attr = Attribute(Series(dates, name='DateTime'))
    indexes = attr.bin_indexes()
    assert len(indexes) == len(dates)


def test_pseudonymize_strings():
    strings = Series(['Abc', 'edf', 'Abc', 'take', '中国', 'edf', 'Abc'])
    attr = Attribute(strings, name='String')
    pseudonyms = attr.pseudonymize()
    assert array_equal(strings.value_counts().values,
                       pseudonyms.value_counts().values)


def test_pseudonymize_ints():
    ints = Series([11, 2, 3, 4, 5, 4, 3, 2, 3, 4, 11])
    attr = Attribute(ints, name='Integer')
    pseudonyms = attr.pseudonymize()
    assert array_equal(ints.value_counts().values,
                       pseudonyms.value_counts().values)


def test_pseudonymize_floats():
    floats = Series([11.5, 2.6, 3.0, 4.3, 5, 4.3, 3.0, 2.6, 3.0, 4.3, 11.6])
    attr = Attribute(floats, name='Float')
    pseudonyms = attr.pseudonymize()
    assert array_equal(floats.value_counts().values,
                       pseudonyms.value_counts().values)


def test_to_pseudonym_dates():
    ints = Series(['07/15/2019', '07/24/2019', '07/23/2019', '07/22/2019',
                   '07/21/2019', '07/22/2019', '07/23/2019', '07/24/2019',
                   '07/23/2019', '07/22/2019', '07/15/2019'])
    attr = Attribute(ints, name='Date')
    pseudonyms = attr.pseudonymize()
    assert array_equal(ints.value_counts().values,
                       pseudonyms.value_counts().values)


def test_random_ints():
    ints = [3, 5, 7, 8, 7, 1, 10, 30, 16, 19]
    attr = Attribute(ints, name='Integer')
    randoms = attr.random()
    assert len(randoms) == len(ints)


def test_random_datetimes():
    datetimes = ['07/15/2019', '07/24/2019', '07/23/2019', '07/22/2019',
                 '07/21/2019', '07/22/2019', '07/23/2019', '07/24/2019',
                 '07/23/2019', '07/22/2019', '07/15/2019']
    attr = Attribute(datetimes, name='Date')
    randoms = attr.random()
    assert len(randoms) == len(datetimes)


def test_random_strings():
    strings = list(map(lambda x: randomize_string(5), range(size)))
    attr = Attribute(Series(strings, name='String'))
    randoms = attr.random()
    assert len(randoms) == size


def test_encode_numerical_attributes():
    from .testdata import adults01
    attr = Attribute(adults01['age'])
    assert attr.bins[0] <= 19
    assert attr.bins[-1] >= 56
    assert len(attr.encode()) == len(attr)

    from sklearn.model_selection import train_test_split
    train, test = train_test_split(adults01['age'])
    assert len(attr.encode(data=train)) == len(train)


def test_encode_categorical_attributes():
    from pandas import DataFrame
    from .testdata import adults01
    frame = DataFrame(adults01)
    attr = Attribute(frame['education'], categorical=True)
    columns = ['11th', '7th-8th', '9th', 'Assoc-acdm', 'Bachelors', 'Doctorate',
               'HS-grad', 'Masters', 'Some-college']
    assert array_equal(attr.bins, columns)
    assert array_equal(attr.encode().columns, columns)


def test_encode_datetime_attributes():
    from pandas import DataFrame
    from .testdata import adults01
    frame = DataFrame(adults01)
    attr = Attribute(frame['birth'])
    # assert other information
    assert len(attr.encode()) == len(attr)

