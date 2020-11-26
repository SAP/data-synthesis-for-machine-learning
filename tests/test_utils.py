# -*- coding: utf-8 -*-
import numpy as np
import pytest

from ds4ml.utils import (plot_histogram,
                         plot_confusion_matrix,
                         plot_heatmap, write_csv, mutual_information,
                         normalize_range, is_datetime, str_to_list,
                         normalize_distribution, has_header,
                         read_data_from_csv, ends_with_json)


def test_plot_confusion_matrix_output_string():
    from pandas import DataFrame
    df = DataFrame({'True': [2, 3], 'False': [5, 0]})
    res = plot_confusion_matrix(df)
    assert type(res) == str
    assert res.startswith('<svg')
    assert res.endswith('</svg>')


@pytest.mark.skip(reason='Need manually test to check figures.')
# Please remove the annotation when manually test
def test_plot_figures_output_show_special_characters():
    bins = np.array(['你好', 'Self-る', '¥¶ĎǨД'])
    counts = np.array([[6, 2, 1], [6, 2, 1]])
    plot_histogram(bins, counts, otype='show')


@pytest.mark.skip(reason='Need manually test to check figures.')
def test_plot_figures_output_show():
    from pandas import DataFrame
    plot_confusion_matrix(DataFrame({'True': [2, 3],
                                     'False': [5, 0]}),
                          otype='show')
    plot_confusion_matrix(DataFrame({'7th-8th': [2, 3, 5, 0],
                                     'Masters': [0, 4, 1, 0],
                                     '11th': [0, 1, 5, 2],
                                     'Bachelors': [2, 0, 0, 6]}),
                          otype='show')

    bins = np.array([28., 29.25, 30.5, 31.75, 33., 34.25, 35.5, 36.75, 38.,
                     39.25, 40.5, 41.75, 43., 44.25, 45.5, 46.75, 48., 49.25,
                     50.5])
    counts = np.array(
        [[1, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
         [1, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
    plot_histogram(bins, counts, otype='show')

    bins = np.array(['Private', 'Self-emp-not-inc', 'State-gov'])
    counts = np.array([[6, 2, 1], [6, 2, 1]])
    plot_histogram(bins, counts, otype='show')

    bins = np.array(['11th', '9th', 'Bachelors', 'HS-grad', 'Masters'])
    counts = np.array([[3, 2, 2, 1, 1], [3, 2, 2, 1, 1]])
    plot_histogram(bins, counts, otype='show')

    bins = np.array([5., 5.45, 5.9, 6.35, 6.8, 7.25, 7.7, 8.15, 8.6,
                     9.05, 9.5, 9.95, 10.4, 10.85, 11.3, 11.75, 12.2, 12.65,
                     13.1])
    counts = np.array(
        [[1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
         [1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0]])
    plot_histogram(bins, counts, otype='show')

    bins = np.array(['Female', 'Male'])
    counts = np.array([[5, 4], [5, 4]])
    plot_histogram(bins, counts, otype='show')

    from .testdata import adults01
    from ds4ml.metrics import pairwise_mutual_information
    data = pairwise_mutual_information(DataFrame(adults01))
    plot_heatmap(data, otype='show')


@pytest.mark.skip(reason='TODO')
def test_mutual_information():
    from pandas import DataFrame
    from .testdata import adults01
    frame = DataFrame(adults01)
    print(mutual_information(frame['age'], frame.drop('age', axis=1)))


def test_write_csv():
    data = [['epsilon', 'c00', 'precision'], [0.2, 157, 0.4]]
    import os
    name = '__test.csv'
    if os.path.exists(name) and os.path.isfile(name):
        os.remove(name)
    write_csv(name, data)
    assert os.path.exists(name)
    assert os.path.isfile(name)
    with open(name, 'r') as file:
        assert file.readline().strip() == 'epsilon,c00,precision'
        assert file.readline().strip() == '0.2,157,0.4'
        file.close()
    os.remove(name)


def test_normalize_range_ints():
    from numpy.random import randint
    for i in range(50):
        start = randint(0, 5)
        stop = randint(start + 1, 200)
        bins = randint(8, 30)
        ints = normalize_range(start, stop, bins)
        assert len(ints) <= bins + 1


def test_normalize_range_floats():
    from numpy.random import randint, rand
    for i in range(50):
        start = round(randint(0, 5) * rand(), 4)
        stop = round(randint(0, 200) * rand(), 4) + 5
        bins = randint(8, 30)
        floats = normalize_range(start, stop, bins)
        assert len(floats) <= bins + 1


def test_is_datetime():
    date = 'monday'
    idt = is_datetime(date)
    assert idt is False
    time = '2020-03-01'
    idt = is_datetime(time)
    assert idt is True
    value = 'high school'
    idt = is_datetime(value)
    assert idt is False


def test_str_to_list():
    iva = '1,3,4,5'
    res = str_to_list(iva)
    assert res == ['1', '3', '4', '5']
    iva = 'name,age,weight,height'
    res = str_to_list(iva)
    assert res == ['name', 'age', 'weight', 'height']


def test_normalize_distribution():
    frequencies = [3, 3, 2]
    res = normalize_distribution(frequencies)
    assert res[0] == 0.375
    assert res[1] == 0.375
    assert res[2] == 0.25


def test_has_header():
    from .testdata import adult_with_head, adult_without_head
    import io
    hasheader = has_header(io.StringIO(adult_with_head))
    assert hasheader is True
    hasheader = has_header(io.StringIO(adult_without_head))
    assert hasheader is False


def test_read_data_from_csv():
    from pandas import DataFrame
    from .testdata import adult_with_head, adult_with_head_res
    import io
    data = read_data_from_csv(io.StringIO(adult_with_head))
    assert data.equals(DataFrame(adult_with_head_res)) is True


def test_ends_with_json():
    assert ends_with_json("d.json") is True
    assert ends_with_json("a.json") is True
    assert ends_with_json("data\ A.jSon") is True
    assert ends_with_json("data A.jSon") is True
