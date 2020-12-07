# -*- coding: utf-8 -*-

"""
Utility functions for data synthesis. Including:
    input/output,
    machine learning,
    ...
"""
import argparse
import csv
import numpy as np
import os
import re
import hashlib

from string import ascii_lowercase
from pandas import Series, DataFrame


# ---------------------------------------------------------------
# Utilities for Input/Output


def has_header(path, encoding='utf-8', sep=','):
    """
    Auto-detect if csv file has header.
    """
    from pandas import read_csv

    def _offset_stream():
        from io import StringIO
        if isinstance(path, StringIO):
            path.seek(0)

    _offset_stream()
    df0 = read_csv(path, header=None, nrows=10, skipinitialspace=True,
                   encoding=encoding, sep=sep)
    _offset_stream()
    df1 = read_csv(path, nrows=10, skipinitialspace=True, encoding=encoding, sep=sep)
    # If the column is numerical, its dtype is different without/with header
    # TODO how about categorical columns
    _offset_stream()
    return tuple(df0.dtypes) != tuple(df1.dtypes)


def ends_with_json(path):
    return re.match(r".+\.json", path, re.IGNORECASE) is not None


def read_data_from_csv(path, na_values=None, header=None, sep=","):
    """
    Read data set from csv or other delimited text file. And remove empty
    columns (all values are null).
    """
    from pandas import read_csv

    try:
        header = header or ('infer' if has_header(path, sep=sep) else None)
        df = read_csv(path, skipinitialspace=True, na_values=na_values,
                      header=header, sep=sep, float_precision='high')
    except (UnicodeDecodeError, NameError):
        header = header or ('infer' if has_header(path, encoding='latin1',
                                                  sep=sep) else None)
        df = read_csv(path, skipinitialspace=True, na_values=na_values,
                      header=header, encoding='latin1', sep=sep,
                      float_precision='high')

    # Remove columns with empty active domain, i.e., all values are missing.
    before_attrs = set(df.columns)
    df.dropna(axis=1, how='all')
    after_attrs = set(df.columns)
    if len(before_attrs) > len(after_attrs):
        print(
            f'Empty columns are removed, include {before_attrs - after_attrs}.')
    # Remove rows with all empty values
    df.dropna(axis=0, how='all')
    if header is None:
        df = df.rename(lambda x: f'#{x}', axis='columns')
    return df


def write_csv(file, data):
    """ write data to csv files """
    # if data is not a list of list, make it:
    if isinstance(data, list) and not isinstance(data[0], list):
        data = [data]
    with open(file, 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)
    f.close()


def file_name(path):
    """ Return the file name without extension from a path. """
    return os.path.splitext(os.path.basename(path))[0]


def str_to_list(val, separator=','):
    """
    Split one string to a list by separator.
    """
    if val is None:
        return None
    return val.split(separator)


# ---------------------------------------------------------------
# Utilities for Plotting


def _compress_svg_data(svg: str):
    value = re.sub(r'\n', ' ', svg)
    value = re.sub(r' {2,}', ' ', value)
    value = re.sub(r'<style type="text/css">.*</style>', '', value)
    value = re.sub(r'<!--(.*?)-->', '', value)
    value = re.sub(r'<\?xml.*\?>', '', value)
    value = re.sub(r'<!DOCTYPE.*dtd">', '', value)
    return value.strip()


def _prepare_for_cjk_characters(chars):
    """ If input string has Chinese, Japanese, and Korean characters, set
    specific font for them. """
    has_cjk = False
    for c in chars:
        if any([start <= ord(c) <= end for start, end in [
            (4352, 4607), (11904, 42191), (43072, 43135), (44032, 55215),
            (63744, 64255), (65072, 65103), (65381, 65500), (131072, 196607)
        ]]):
            has_cjk = True
            break
    if has_cjk:
        import matplotlib
        matplotlib.rcParams['font.family'] = ['Microsoft Yahei']


def plot_confusion_matrix(matrix: DataFrame, title='',
                          ylabel='Predicted', xlabel='Actual',
                          otype='string', path=None, cmap='Blues',
                          vrange=None):
    """
    Plot a confusion matrix to show predict and actual values.

    Parameters
    ----------
    matrix : pandas.DataFrame
        the matrix to plot

    title : str
        title of image

    ylabel : str
        label in y-axis of image

    xlabel : str
        label in x-axis of image

    otype : str
        output type, support 'string' (default), 'file', 'show'

    path : str
        output path for 'file' type, default is 'matrix_2_3.svg' ((2, 3) is the
        shape of matrix).

    cmap : str
        color

    vrange : 2-tuple
        manually set range of matrix
    """
    import matplotlib.pyplot as plt
    _prepare_for_cjk_characters(''.join(map(str, matrix.columns)))
    figsize = (6.4, 4.8)
    n_columns = matrix.columns.size
    if n_columns > 9:
        from math import ceil
        width = 6.4 + ceil((n_columns - 9) / 2) * 0.8
        height = width * figsize[1] / figsize[0]
        figsize = (width, height)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if vrange is None:
        vrange = (matrix.values.min(), matrix.values.max())
    exp = len(str(vrange[1])) - 1
    if exp > 2:
        matrix = matrix.div(10 ** exp)
        vrange = (vrange[0] / (10 ** exp), vrange[1] / (10 ** exp))
    im = ax.imshow(matrix.values, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
    ax.set(xticks=np.arange(matrix.shape[1]),
           yticks=np.arange(matrix.shape[0]),
           xticklabels=matrix.columns,
           yticklabels=matrix.index,
           title=title,
           ylabel=ylabel,
           xlabel=xlabel)
    ax.tick_params(which='both', length=0)
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    # Loop over data dimensions and create text annotations.
    thresh = (vrange[0] + vrange[1]) / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if exp > 2:
                text = '{:.2f}'.format(matrix.iloc[i, j])
            else:
                text = matrix.iloc[i, j]
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if matrix.iloc[i, j] > thresh else "black")
    from ds4ml.metrics import error_rate
    ax.set_title('Rate: {:.1%}'.format(error_rate(matrix)), fontsize=10)

    cbar = ax.figure.colorbar(im, ax=ax, pad=0.03, aspect=30)
    cbar.outline.set_visible(False)
    if exp > 2:
        cbar.ax.text(0, -0.1, f'(e+{exp})')
    cbar.ax.tick_params(which='both', length=0)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.subplots_adjust()
    try:
        if otype == 'string':
            from io import StringIO
            img = StringIO()
            plt.savefig(img, format='svg', bbox_inches='tight')
            return _compress_svg_data(img.getvalue())
        elif otype == 'file':
            if path is None:
                path = 'matrix_{}_{}.svg'.format(matrix.shape[0],
                                                 matrix.shape[1])
            plt.savefig(path, bbox_inches='tight')
            return path
        elif otype == 'show':
            plt.show()
    finally:
        plt.close()


def plot_histogram(bins, heights, otype='string', path=None,
                   x_label='', y_label='Frequency'):
    """
    Plot two bars.
    Note: Because pyplot draws diagrams auto-scale, this function will provide
    kinds of diagram patterns.
    """
    import matplotlib.pyplot as plt
    _prepare_for_cjk_characters(''.join(map(str, bins)))
    length = len(bins)
    fig_width = 6.4
    ticks = 22
    if length < 5:
        # make 7 bars to show the data
        # insert 0 or '' to the center of array
        # e.g. [3, 4] => [3, 0, 4], ['1', '2', '3'] => ['1', '', '2', '', '3'], ...
        bins = list(map(str, bins))
        bins = ',,'.join(bins).split(',')
        heights = np.insert(heights, list(range(1, length)), 0, axis=1)
        # pad 0 or '' to array in the begin and end
        pad = (7 - len(bins)) // 2
        bins = tuple([''] * pad + bins + [''] * pad)
        heights = np.append(np.insert(heights, [0], [0] * pad, axis=1),
                            np.zeros((len(heights), pad), dtype=int), axis=1)
        length = 7
    else:
        # TODO: if count of bins is bigger than 33, and it is categorical, how?
        if length >= 60:
            bins = bins[:60]
            heights = heights[:, :60]
            length = 60
        bins = tuple(map(str, bins))
        length_ = [8, 12, 16, 20, 32, 42, 48, 60]
        ticks_ = [22, 30, 36, 48, 76, 96, 108, 172]
        width_ = [6.4, 9.6, 11.2, 11.2, 11.2, 14, 14, 15]
        idx = len([i for i in length_ if length > i])
        ticks = ticks_[idx]
        fig_width = width_[idx]

    x = np.arange(length)
    fig = plt.figure(figsize=(fig_width, 4.8))
    ax = fig.add_subplot(111)
    width = length / ticks
    diff = width / 2 + 0.01
    ax.bar(x - diff, heights[0], width=width, color='#38bbe8')
    ax.bar(x + diff, heights[1], width=width, color='#ffd56e')
    ax.legend(['raw', 'synth'])
    fig.autofmt_xdate()
    # fig.tight_layout(pad=1.5)
    plt.xticks(x, bins, fontsize=10)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.subplots_adjust()
    try:
        if otype == 'string':
            from io import StringIO
            img = StringIO()
            plt.savefig(img, format='svg', bbox_inches='tight')
            return _compress_svg_data(img.getvalue())
        elif otype == 'file':
            if path is None:
                path = 'histogram_{}_{}.svg'.format(len(bins), len(heights))
            plt.savefig(path, bbox_inches='tight')
            return path
        elif otype == 'show':
            plt.show()
    finally:
        plt.close()


def plot_heatmap(data, title='', otype='string', path=None, cmap='Blues'):
    """
    Plot a heatmap to show pairwise correlations (e.g. mutual information).

    Parameters see <code>plot_confusion_matrix</code>
    """
    import matplotlib.pyplot as plt
    _prepare_for_cjk_characters(''.join(data.columns))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(data.values, cmap=cmap)

    ax.set_title(title, fontsize=10)
    ticks = np.arange(len(data.columns))
    ax.set_xticks(ticks)
    ax.set_xticklabels(data.columns)
    ax.set_yticks(ticks)
    ax.set_yticklabels(data.index)
    ax.tick_params(which='both', length=0)
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    # set color bar in the right
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.03, aspect=30)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(which='both', length=0)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.subplots_adjust()
    try:
        if otype == 'string':
            from io import StringIO
            img = StringIO()
            plt.savefig(img, format='svg', facecolor='#ebebeb',
                        bbox_inches='tight')
            return _compress_svg_data(img.getvalue())
        elif otype == 'file':
            if path is None:
                path = 'heatmap_{}_{}.svg'.format(data.shape[0], data.shape[1])
            plt.savefig(path, facecolor='#ebebeb', bbox_inches='tight')
            return path
        elif otype == 'show':
            plt.show()
    finally:
        plt.close()


# ---------------------------------------------------------------
# Utilities for Metrics, ML

def train_and_predict(x_train, y_train, x_test):
    """
    Predict <x, y> by SVM classifier, and compare with test data
    """
    from sklearn.svm import SVC
    classifier = SVC(gamma='scale')
    classifier.fit(x_train, y_train)
    result = classifier.predict(x_test)
    return result


def mutual_information(child: Series, parents: DataFrame):
    """
    Mutual information of child (Series) and parents (DataFrame) distributions
    """
    from sklearn.metrics import mutual_info_score
    if parents.shape[1] == 1:
        parents = parents.iloc[:, 0]
    else:
        parents = parents.apply(lambda x: ' '.join(x.array), axis=1)
    return mutual_info_score(child, parents)


def normalize_distribution(frequencies):
    frequencies = np.array(frequencies, dtype=float)
    frequencies = frequencies.clip(0)
    total = frequencies.sum()
    if total > 0:
        if np.isinf(total):
            return normalize_distribution(np.isinf(frequencies))
        else:
            return frequencies / total
    else:
        return np.full_like(frequencies, 1 / frequencies.size)


def normalize_range(start, stop, bins=20):
    """
    Return evenly spaced values within a given interval, and a dynamically
    calculated step, to make number of bins close to 20. If integer interval,
    the smallest step is 1; If float interval, the smallest step is 0.5.
    """
    from math import ceil, floor
    if isinstance(start, int) and isinstance(stop, int):
        step = ceil((stop - start) / bins)
    else:
        start = floor(start)
        stop = ceil(stop)
        step = (stop - start) / bins
        step = ceil(step) if ceil(step) == round(step) else round(step) + 0.5
    stop = step * (bins + 1) + start
    return np.arange(start, stop, step)


def is_datetime(value: str):
    from dateutil.parser import parse
    """
    Detect a value is a datetime. Exclude some datetime literals (weekdays and
    months) from method `dateutil.parser`.
    """
    literals = {'mon', 'monday', 'tue', 'tuesday', 'wed', 'wednesday', 'thu',
                'thursday', 'fri', 'friday', 'sat', 'saturday', 'sun', 'sunday',
                'jan', 'january', 'feb', 'february', 'mar', 'march', 'apr',
                'april', 'may', 'jun', 'june', 'jul', 'july', 'aug', 'august',
                'sep', 'sept', 'september', 'oct', 'october', 'nov', 'november',
                'dec', 'december'}
    try:
        value = value.lower()
        if value in literals:
            return False
        parse(value)
        return True
    except ValueError:
        return False
    except AttributeError:
        return False


def randomize_string(length):
    return ''.join(np.random.choice(list(ascii_lowercase), size=length))


def pseudonymise_string(value):
    """ pseudonymise a string by RIPEMD-160 hashes """
    return hashlib.new('ripemd160', str(value).encode('utf-8')).hexdigest()


# ---------------------------------------------------------------
# Utilities for arguments parser


class CustomFormatter(argparse.HelpFormatter):
    def _format_action_invocation(self, action):
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar
        else:
            parts = []
            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)

            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            # change to
            #    -s, --long ARGS
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    # parts.append('%s %s' % (option_string, args_string))
                    parts.append('%s' % option_string)
                parts[-1] += ' %s' % args_string
            return ', '.join(parts)
