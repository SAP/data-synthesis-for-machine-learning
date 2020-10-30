"""
Attribute: data structure for 1-dimensional cross-sectional data
"""

import numpy as np

from bisect import bisect_right
from random import uniform
from pandas import Series, DataFrame
from dateutil.parser import parse
from datetime import datetime, timedelta

import ds4ml
from ds4ml import utils


class Attribute(Series):

    _epoch = datetime(1970, 1, 1)  # for datetime handling

    def __init__(self, data, name=None, dtype=None, index=None, copy=False,
                 fastpath=False, categorical=False):
        """
        A Series with extra information, e.g. categorical.

        Parameters
        ----------
        categorical : bool
            set categorical label for attribute. If categorical, this attribute
            takes on a limited and fixed number of possible values. Examples:
            blood type, gender.
        """
        Series.__init__(self, data, name=name, dtype=dtype, index=index,
                        copy=copy, fastpath=fastpath)

        # bins can be int (size of histogram bins), str (as algorithm name),
        self._bins = ds4ml.params['attribute.bins']

        self._min = None
        self._max = None
        self._step = None

        # probability distribution (pr)
        self.bins = None
        self.prs = None

        from pandas.api.types import infer_dtype
        # atype: date type for handle different kinds of attributes in data
        # synthesis, support: integer, float, string, datetime.
        self.atype = infer_dtype(self, skipna=True)
        if self.atype == 'integer':
            pass
        elif self.atype == 'floating' or self.atype == 'mixed-integer-float':
            self.atype = 'float'
        elif self.atype in ['string', 'mixed-integer', 'mixed']:
            self.atype = 'string'
            if all(map(utils.is_datetime, self._values)):
                self.atype = 'datetime'

        # fill the missing values with the most frequent value
        self.fillna(self.mode()[0], inplace=True)

        # special handling for datetime attribute
        if self.atype == 'datetime':
            self.update(self.map(self._to_seconds).map(self._date_formatter))

        if self.atype == 'float':
            self._decimals = self.decimals()

        # how to define the attribute is categorical.
        self.categorical = categorical or (
                self.atype == 'string' and not self.is_unique)
        self._set_domain()
        self._set_distribution()

    # handling functions for datetime attribute
    def _to_seconds(self, timestr):
        return int((parse(timestr) - self._epoch).total_seconds())

    def _date_formatter(self, seconds):
        date = self._epoch + timedelta(seconds=seconds)
        return '%d/%d/%d' % (date.month, date.day, date.year)

    # Take pandas.Series as manipulation result.
    @property
    def _constructor(self):
        return Series

    @property
    def _constructor_expanddim(self):
        from ds4ml.dataset import DataSet
        return DataSet

    @property
    def is_numerical(self):
        return self.atype == 'integer' or self.atype == 'float'

    @property
    def domain(self):
        """
        Return attribute's domain, which can be a list of values for categorical
        attribute, and an interval with min/max value for non-categorical
        attribute.
        """
        if self.categorical:
            return self.bins
        else:
            return [self._min, self._max]

    @domain.setter
    def domain(self, domain: list):
        """
        Set attribute's domain, includes min, max, frequency, or distribution.

        Generally, the domain of one attribute can be calculated automatically.
        This method can be manually called for specific purposes, e.g. compare
        two same attributes based on same domain.

        Parameters
        ----------
        domain : list
            domain of one attribute. For numerical or datetime attributes, it
            should be a list of two elements [min, max]; For categorical
            attributes, it should a list of potential values of this attribute.
        """
        # if a attribute is numerical and categorical and domain's length is
        # bigger than 2, take it as categorical. e.g. zip code.
        if self.atype == 'datetime':
            domain = list(map(self._to_seconds, domain))
        if (self.is_numerical and self.categorical and len(domain) > 2) or (
                self.categorical):
            self._min = min(domain)
            self._max = max(domain)
            self.bins = np.array(domain)
        elif self.is_numerical:
            self._min, self._max = domain
            self._step = (self._max - self._min) / self._bins
            self.bins = np.array([self._min, self._max])
        elif self.atype == 'string':
            lengths = [len(str(i)) for i in domain]
            self._min = min(lengths)
            self._max = max(lengths)
            self.bins = np.array(domain)
        self._set_distribution()

    def _set_domain(self):
        """
        Compute domain (min, max, distribution bins) from input data
        """
        if self.atype == 'string':
            self._items = self.astype(str).map(len)
            self._min = int(self._items.min())
            self._max = int(self._items.max())
            if self.categorical:
                self.bins = self.unique()
            else:
                self.bins = np.array([self._min, self._max])
        elif self.atype == 'datetime':
            self.update(self.map(self._to_seconds))
            if self.categorical:
                self.bins = self.unique()
            else:
                self._min = float(self.min())
                self._max = float(self.max())
                self.bins = np.array([self._min, self._max])
                self._step = (self._max - self._min) / self._bins
        else:
            self._min = float(self.min())
            self._max = float(self.max())
            if self.categorical:
                self.bins = self.unique()
            else:
                self.bins = np.array([self._min, self._max])
                self._step = (self._max - self._min) / self._bins

    def _set_distribution(self):
        if self.categorical:
            counts = self.value_counts()
            for value in set(self.bins) - set(counts.index):
                counts[value] = 0
            counts.sort_index(inplace=True)
            if self.atype == 'datetime':
                counts.index = list(map(self._date_formatter, counts.index))
            self._counts = counts.values
            self.prs = utils.normalize_distribution(counts)
            self.bins = np.array(counts.index)
        else:
            # Note: hist, edges = numpy.histogram(), all but the last bin
            # is half-open. If bins is 20, then len(hist)=20, len(edges)=21
            if self.atype == 'string':
                hist, edges = np.histogram(self._items,
                                           bins=self._bins)
            else:
                hist, edges = np.histogram(self, bins=self._bins,
                                           range=(self._min, self._max))
            self.bins = edges[:-1]  # Remove the last bin edge
            self._counts = hist
            self.prs = utils.normalize_distribution(hist)
            if self.atype == 'integer':
                self._min = int(self._min)
                self._max = int(self._max)

    def counts(self, bins=None, normalize=True):
        """
        Return an array of counts (or normalized density) of unique values.

        This function works with `attribute.bins`. Combination of both are
        like `Series.value_counts`. The parameter `bins` can be none, or a list.
        """
        if bins is None:
            return self._counts
        if self.categorical:
            if self.atype == 'datetime':
                bins = list(map(self._to_seconds, bins))
            counts = self.value_counts()
            for value in set(bins) - set(counts.index):
                counts[value] = 0
            if normalize:
                return np.array([round(counts.get(b)/sum(counts) * 100, 2)
                                 for b in bins])
            else:
                return np.array([counts.get(b) for b in bins])
        else:
            if len(bins) == 1:
                return np.array([self.size])
            hist, _ = np.histogram(self, bins=bins)
            if normalize:
                return (hist / hist.sum() * 100).round(2)
            else:
                return hist

    def bin_indexes(self):
        """
        Encode values into bin indexes for Bayesian Network.
        """
        if self.categorical:
            mapping = {value: idx for idx, value in enumerate(self.bins)}
            indexes = self.map(lambda x: mapping[x], na_action='ignore')
        else:
            indexes = self.map(lambda x: bisect_right(self.bins, x) - 1,
                               na_action='ignore')
        indexes.fillna(len(self.bins), inplace=True)
        return indexes.astype(int, copy=False)

    def metadata(self):
        """
        Return attribution's metadata information in JSON format or Python
        dictionary. Usually used in debug and testing.
        """
        return {
            'name': self.name,
            'dtype': self.dtype,
            'atype': self.atype,
            'categorical': self.categorical,
            'min': self._min,
            'max': self._max,
            'bins': self.bins.tolist(),
            'prs': self.prs.tolist()
        }

    def decimals(self):
        """
        Returns number of decimals places for floating attribute.
        """
        def decimals_of(value: float):
            value = str(value)
            return len(value) - value.rindex('.') - 1

        vc = self.map(decimals_of).value_counts()
        slot = 0
        for i in range(len(vc)):
            if sum(vc.head(i + 1)) / sum(vc) > 0.8:
                slot = i + 1
                break
        return max(vc.index[:slot])

    def pseudonymize(self):
        """
        Return pseudonymized values for this attribute, which is used to
        substitute identifiable data with a reversible, consistent value.
        """
        if self.categorical:
            mapping = {b: utils.pseudonymise_string(b) for b in self.bins}
            return self.map(lambda x: mapping[x])

        if self.atype == 'string':
            return self.map(utils.pseudonymise_string)
        elif self.is_numerical or self.atype == 'datetime':
            return self.map(str).map(utils.pseudonymise_string)

    def random(self, size=None):
        """
        Return an random array with same length (usually used for
        non-categorical attribute).
        """
        if size is None:
            size = len(self)
        if self._min == self._max:
            rands = np.ones(size) * self._min
        else:
            rands = np.arange(self._min, self._max,
                              (self._max - self._min) / size)

        np.random.shuffle(rands)
        if self.atype == 'string':
            if self._min == self._max:
                length = self._min
            else:
                length = np.random.randint(self._min, self._max)
            vectorized = np.vectorize(lambda x: utils.randomize_string(length))
            rands = vectorized(rands)
        elif self.atype == 'integer':
            rands = list(map(int, rands))
        elif self.atype == 'datetime':
            rands = list(map(self._date_formatter, rands))
        return Series(rands)

    def _sampling_bins(self, index: int):
        if self.categorical:
            return self.bins[index]

        length = len(self.bins)
        if index < length - 1:
            return uniform(self.bins[index], self.bins[index + 1])
        else:
            return uniform(self.bins[-1], self._max)

    def choice(self, size=None, indexes=None):
        """
        Return a random sample based on this attribute's probability.
        If indexes and n are both set, ignore n.

        Parameters
        ----------
        size : int
            size of random sample

        indexes : array-like
            array of indexes in bins
        """
        if indexes is None:
            if size is None:
                size = len(self)
            indexes = Series(np.random.choice(len(self.prs),
                                              size=size, p=self.prs))
        column = indexes.map(lambda x: self._sampling_bins(x))
        if self.atype == 'datetime':
            if not self.categorical:
                column = column.map(self._date_formatter)
        elif self.atype == 'float':
            column = column.round(self._decimals)
        elif self.atype == 'integer':
            column = column.round().astype(int)
        elif self.atype == 'string':
            if not self.categorical:
                column = column.map(lambda x: utils.randomize_string(int(x)))
        return column

    def encode(self, data=None):
        """
        Encode labels to normalized encoding.

        Parameters
        ----------
        data : array-like
            target values
        """
        if data is None:
            data = self.copy()
        else:
            if self.atype == 'datetime':
                if all(map(utils.is_datetime, data)):
                    data = data.map(self._to_seconds)
                else:
                    data = data.map(int)

        if self.categorical:
            df = DataFrame()
            for c in self.bins:
                df[c] = data.apply(lambda v: 1 if v == c else 0)
            return df

        if self.atype != 'string':
            return data.apply(lambda v:  # 1e-8 is a small delta
                              int((v - self._min) / (self._step + 1e-8))
                              / self._bins)
        else:
            raise ValueError('Non-categorical attribute does not need encode '
                             'method.')
