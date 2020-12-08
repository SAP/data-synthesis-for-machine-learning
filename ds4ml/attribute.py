"""
Attribute: data structure for 1-dimensional cross-sectional data

This class only handle integer, float, string, datetime columns, and it can be
labeled as categorical column.
"""
from bisect import bisect_right
from random import uniform
from pandas import Series, DataFrame
from dateutil.parser import parse
from datetime import datetime, timedelta

import numpy as np

from ds4ml import utils


# Default environment variables for data processing and analysis
DEFAULT_BIN_SIZE = 20


class AttributePattern:
    """
    A helper class of ``Attribute`` to store its patterns.
    """
    # _type: date type for handle different kinds of attributes in data
    # synthesis, only support: integer, float, string, datetime.
    _type = None
    categorical = False
    # min, max has been defined as member function of pandas.Series
    min_ = None
    max_ = None
    _decimals = None

    # probability distribution (pr)
    bins = None
    prs = None
    _counts = None
    _pattern_generated = False

    # Here _bin_size is int-typed (to show the size of histogram bins), which
    # is different from bins in np.histogram.
    _bin_size = DEFAULT_BIN_SIZE

    @property
    def type(self):
        return self._type


class Attribute(AttributePattern, Series):

    _epoch = datetime(1970, 1, 1)  # for datetime handling

    def __init__(self, *args, **kwargs):
        """
        An improved Series with extra pattern information, e.g. categorical,
        min/max value, and probability distribution.

        The ``Attribute`` class has two modes:

        - it has raw data, and then can calculate its pattern from the data;

        - it doesn't have raw data, and only have the pattern from customer.

        Parameters
        ----------
        categorical : bool
            set categorical label for attribute. If categorical, this attribute
            takes on a limited and fixed number of possible values. Examples:
            blood type, gender.
        """
        categorical = kwargs.pop('categorical', False)
        super().__init__(*args, **kwargs)
        self.set_pattern(categorical=categorical)

    def _calculate_pattern(self):
        from pandas.api.types import infer_dtype
        self._type = infer_dtype(self, skipna=True)
        if self._type == 'integer':
            pass
        elif self._type == 'floating' or self._type == 'mixed-integer-float':
            self._type = 'float'
        elif self._type in ['string', 'mixed-integer', 'mixed']:
            self._type = 'string'
            if all(map(utils.is_datetime, self._values)):
                self._type = 'datetime'

        # fill the missing values with the most frequent value
        self.fillna(self.mode()[0], inplace=True)

        # for datetime attribute is converted to seconds since Unix epoch time
        if self.type == 'datetime':
            self.update(self.map(self._to_seconds))

        if self.type == 'float':
            self._decimals = self.decimals()

        # The `categorical` option can be set to true when the attribute is
        # string-typed and all values are not unique, and its value can be
        # overrode by user.
        self.categorical = self.categorical or (
            self.type == 'string' and not self.is_unique)
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

    def set_pattern(self, pattern=None, **kwargs):
        """
        Set an attribute's pattern, including its type, min/max value, and
        probability distributions.
        If patter is None, then calculation its pattern from its data.
        """
        if not self._pattern_generated:
            self.categorical = kwargs.pop("categorical", False)
            if pattern is None:
                # to calculate the pattern use its data
                self._calculate_pattern()
            else:
                self._type = pattern['type']
                if self.type == 'float':
                    self._decimals = pattern['decimals']
                self.categorical = pattern['categorical']
                self.min_ = pattern['min']
                self.max_ = pattern['max']
                self.bins = np.array(pattern['bins'])
                self.prs = np.array(pattern['prs'])
            self._pattern_generated = True

    @property
    def is_numerical(self):
        return self._type == 'integer' or self._type == 'float'

    @property
    def domain(self):
        """
        Return attribute's domain, which can be a list of values for categorical
        attribute, and an interval with min/max value for non-categorical
        attribute.
        """
        if self.categorical:
            return self.bins
        return [self.min_, self.max_]

    def _step(self):
        """ Return step for numerical or datetime attribute. """
        return (self.max_ - self.min_) / self._bin_size

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
        if self.type == 'datetime':
            domain = list(map(self._to_seconds, domain))
        if (self.is_numerical and self.categorical and len(domain) > 2) or (
                self.categorical):
            self.min_, self.max_ = min(domain), max(domain)
            self.bins = np.array(domain)
        elif self.is_numerical:
            self.min_, self.max_ = domain
            self.bins = np.array([self.min_, self.max_])
        elif self._type == 'string':
            lengths = [len(str(i)) for i in domain]
            self.min_, self.max_ = min(lengths), max(lengths)
            self.bins = np.array(domain)
        self._set_distribution()

    def _set_domain(self):
        """
        Compute domain (min, max, distribution bins) from input data
        """
        if self.categorical:
            self.bins = self.unique()

        if self._type == 'string':
            items = self.astype(str).map(len)
            self.min_ = int(items.min())
            self.max_ = int(items.max())
            if not self.categorical:
                self.bins = np.array([self.min_, self.max_])
        elif self._type == 'datetime':
            if not self.categorical:
                self.min_ = float(self.min())
                self.max_ = float(self.max())
                self.bins = np.array([self.min_, self.max_])
        else:
            self.min_ = float(self.min())
            self.max_ = float(self.max())
            if not self.categorical:
                self.bins = np.array([self.min_, self.max_])

    def _set_distribution(self):
        if self.categorical:
            counts = self.value_counts()
            for value in set(self.bins) - set(counts.index):
                counts[value] = 0
            counts.sort_index(inplace=True)
            if self.type == 'datetime':
                counts.index = list(map(self._date_formatter, counts.index))
            self._counts = counts.values
            self.prs = utils.normalize_distribution(counts)
            self.bins = np.array(counts.index)
        else:
            # Note: hist, edges = numpy.histogram(), all but the last bin
            # is half-open. If bins is 20, then len(hist)=20, len(edges)=21
            if self.type == 'string':
                hist, edges = np.histogram(self.astype(str).map(len),
                                           bins=self._bin_size)
            else:
                hist, edges = np.histogram(self, bins=self._bin_size,
                                           range=(self.min_, self.max_))
            self.bins = edges[:-1]  # Remove the last bin edge
            self._counts = hist
            self.prs = utils.normalize_distribution(hist)
            if self.type == 'integer':
                self.min_ = int(self.min_)
                self.max_ = int(self.max_)

    def counts(self, bins=None, normalize=True):
        """
        Return an array of counts (or normalized density) of unique values.

        This function works with `attribute.bins`. Combination of both are
        like `Series.value_counts`. The parameter `bins` can be none, or a list.
        """
        if bins is None:
            return self._counts
        if self.categorical:
            if self.type == 'datetime':
                bins = list(map(self._to_seconds, bins))
            counts = self.value_counts()
            for value in set(bins) - set(counts.index):
                counts[value] = 0
            if normalize:
                return np.array([round(counts.get(b)/sum(counts) * 100, 2)
                                 for b in bins])
            return np.array([counts.get(b) for b in bins])

        if len(bins) == 1:
            return np.array([self.size])
        hist, _ = np.histogram(self, bins=bins)
        if normalize:
            return (hist / hist.sum() * 100).round(2)
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

    def to_pattern(self):
        """
        Return attribution's metadata information in JSON format or Python
        dictionary. Usually used in debug and testing.
        """
        return {
            'name': self.name,
            'type': self._type,
            'categorical': self.categorical,
            'min': self.min_,
            'max': self.max_,
            'decimals': self._decimals if self.type == 'float' else None,
            'bins': self.bins.tolist(),
            'prs': self.prs.tolist()
        }

    def decimals(self):
        """
        Returns number of decimals places for floating attribute. Used for
        generated dataset to keep consistent decimal places for float attribute.
        """
        def decimals_of(value: float):
            value = str(value)
            return len(value) - value.rindex('.') - 1

        counts = self.map(decimals_of).value_counts()
        slot = 0
        for i in range(len(counts)):
            if sum(counts.head(i + 1)) / sum(counts) > 0.8:
                slot = i + 1
                break
        return max(counts.index[:slot])

    def pseudonymize(self, size=None):
        """
        Return pseudonymized values for this attribute, which is used to
        substitute identifiable data with a reversible, consistent value.
        """
        size = size or self.size
        if size != self.size:
            attr = Series(np.random.choice(self.bins, size=size, p=self.prs))
        else:
            attr = self
        if self.categorical:
            mapping = {b: utils.pseudonymise_string(b) for b in self.bins}
            return attr.map(lambda x: mapping[x])

        if self.type == 'string':
            return attr.map(utils.pseudonymise_string)
        elif self.is_numerical or self.type == 'datetime':
            return attr.map(str).map(utils.pseudonymise_string)

    def random(self, size=None):
        """
        Return an random array with same length (usually used for
        non-categorical attribute).
        """
        size = size or self.size
        if self.min_ == self.max_:
            rands = np.ones(size) * self.min_
        else:
            rands = np.arange(self.min_, self.max_, (self.max_-self.min_)/size)

        np.random.shuffle(rands)
        if self.type == 'string':
            if self.min_ == self.max_:
                length = self.min_
            else:
                length = np.random.randint(self.min_, self.max_)
            vectorized = np.vectorize(lambda x: utils.randomize_string(length))
            rands = vectorized(rands)
        elif self.type == 'integer':
            rands = list(map(int, rands))
        elif self.type == 'datetime':
            rands = list(map(self._date_formatter, rands))
        return Series(rands)

    def retain(self, size=None):
        """ Return retained attribute with the size """
        size = size or self.size
        if size < self.size:
            return self.head(size)
        if size == self.size:
            return self
        copies = size // self.size
        remainder = size - (copies * self.size)

        return Series(self.tolist() * copies + self.head(remainder).tolist())

    def _random_sample_at(self, index: int):
        """ Sample a value from distribution bins at position 'index'"""
        if self.categorical:
            return self.bins[index]

        length = len(self.bins)
        if index < length - 1:
            return uniform(self.bins[index], self.bins[index + 1])
        return uniform(self.bins[-1], self.max_)

    def choice(self, size=None, indexes=None):
        """
        Return a random sample based on this attribute's probability and
        distribution bins (default value is base random distribution bins based
        on its probability).

        Parameters
        ----------
        size : int
            size of random sample

        indexes : array-like
            array of indexes in distribution bins
        """
        if indexes is None:
            size = size or self.size
            indexes = Series(np.random.choice(len(self.prs),
                                              size=size, p=self.prs))
        column = indexes.map(self._random_sample_at)
        if self.type == 'datetime':
            if not self.categorical:
                column = column.map(self._date_formatter)
        elif self.type == 'float':
            column = column.round(self._decimals)
        elif self.type == 'integer':
            column = column.round().astype(int)
        elif self.type == 'string':
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
            if self.type == 'datetime':
                if all(map(utils.is_datetime, data)):
                    data = data.map(self._to_seconds)
                else:
                    data = data.map(int)

        if self.categorical:
            frame = DataFrame()
            for col in self.bins:
                frame[col] = data.apply(lambda v: 1 if v == col else 0)
            return frame

        if self.type != 'string':
            step = self._step()
            return data.apply(lambda v:  # 1e-8 is a small delta
                              int((v - self.min_) / (step + 1e-8))
                              / self._bin_size)
        raise ValueError('Can\'t encode Non-categorical attribute.')
