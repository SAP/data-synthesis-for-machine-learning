"""
DataSet: data structure for potentially mixed-type Attribute.
"""

import json

from pandas import DataFrame, Series

from ds4ml.attribute import Attribute
from ds4ml.synthesizer import greedy_bayes, noisy_conditionals, noisy_distributions
from ds4ml.utils import normalize_distribution


class DataSetPattern:
    """
    A helper class of ``DataSet`` to store its patterns.
    """
    # DataSet's pattern data has the following members:
    _network = None
    _cond_prs = None
    _attrs = None
    _records = None

    # Options of DataSet constructor to preset some properties:
    _categories = []    # categorical columns setting from command lines
    _config = None      # configurations for data-pattern command
    _pattern_generated = False


class DataSet(DataSetPattern, DataFrame):

    def __init__(self, *args, **kwargs):
        """
        An improved DataFrame with extra patterns information, e.g. its bayesian
        network structure, conditional probabilities on the network, and pattern
        information of all its columns.

        The ``DataSet`` class has two modes:

        - it has raw data, and then can calculate its pattern from the data;

        - it doesn't have raw data, and only have the pattern from customer.

        Parameters
        ----------
        categories : list of columns (optional)
            Column names whose values are categorical.
        """
        categories = kwargs.pop("categories", [])
        self._categories = [] if categories is None else categories
        pattern = kwargs.pop('pattern', None)
        super(DataSet, self).__init__(*args, **kwargs)
        self.separator = '_'
        if pattern is not None and all(k in pattern for k in
                                       ['network', 'prs', 'attrs', 'records']):
            self._set_pattern(pattern)
        else:
            self._records = self.shape[0]

    @property
    def _constructor(self):
        return DataSet

    # disable _constructor_sliced method for single column slicing. Try to
    # use __getitem__ method.
    # @property
    # def _constructor_sliced(self):
    #     return Attribute

    def __getitem__(self, key):
        result = super(DataSet, self).__getitem__(key)
        if isinstance(result, Series):
            result.__class__ = Attribute
            if self._attrs is not None:
                result.set_pattern(self._attrs.get(key),
                                   categorical=key in self._categories)
            else:
                result.set_pattern(categorical=key in self._categories)
        return result

    @classmethod
    def from_pattern(cls, pattern: dict):
        """
        Alternate constructor to create a ``DataSet`` from a pattern file.
        """
        # set columns to DataSet, which will set column name to each Attribute.
        columns = pattern['attrs'].keys()
        dataset = DataSet(columns=columns, pattern=pattern)
        return dataset

    @classmethod
    def from_pattern_file(cls, filename):
        with open(filename) as f:
            pattern = json.load(f)
        return cls.from_pattern(pattern)

    def _set_pattern(self, pattern=None):
        """ Set pattern data for the DataSet. """
        if not self._pattern_generated:
            self._network = pattern['network']
            self._cond_prs = pattern['prs']
            self._attrs = pattern['attrs']
            self._config = pattern['config']
            self._records = pattern['records']
            self._pattern_generated = True

    def mi(self):
        """ Return mutual information of pairwise attributes. """
        from ds4ml.metrics import pairwise_mutual_information
        return pairwise_mutual_information(self)

    def encode(self, data=None):
        """
        Transform data set to values by kinds of encoders.
        If data is set, use this data set's encoders to transform.
        """
        # If the data to encode is None, then transform source data _data;
        frame = DataFrame()
        for col in self.columns:
            attr = self[col]
            if data is not None and col not in data:
                continue
            # when attribute is string-typed but not categorical, ignore its
            # encode method.
            if attr.categorical:
                subs = attr.encode(None if data is None else data[col])
                for label in attr.bins:
                    frame[col + self.separator + str(label)] = subs[label]
            elif attr.type != 'string':
                frame[col] = attr.encode(None if data is None else data[col])
        return frame

    def _sampling_dataset(self, network, cond_prs, n):
        """
        Returns a sampling dataset (n rows) based on bayesian network and
        conditional probability.
        """
        from numpy import random
        root_col = network[0][1][0]
        root_prs = cond_prs[root_col]

        columns = [root_col]  # columns from bayesian network
        for node, _ in network:
            columns.append(node)

        frame = DataFrame(columns=columns)  # encoded DataFrame
        frame[root_col] = random.choice(len(root_prs), size=n, p=root_prs)

        for child, parents in network:
            child_cond_prs = cond_prs[child]
            for indexes in child_cond_prs.keys():
                prs = child_cond_prs[indexes]
                indexes = list(eval(indexes))

                filters = ''
                for parent, value in zip(parents, indexes):
                    filters += f"(frame['{parent}']=={value})&"
                filters = eval(filters[:-1])
                size = frame[filters].shape[0]
                if size:
                    frame.loc[filters, child] = random.choice(len(prs),
                                                              size=size,
                                                              p=prs)
            child_prs = self[child].prs
            frame.loc[frame[child].isnull(), child] = random.choice(
                len(child_prs), size=frame[child].isnull().sum(), p=child_prs)
        frame[frame.columns] = frame[frame.columns].astype(int)
        return frame

    def _construct_bayesian_network(self, epsilon=0.1, degree=2,
                                    pseudonyms=None, deletes=None, retains=None):
        """
        Construct bayesian network of the DataSet.
        """
        deletes = deletes or []
        pseudonyms = pseudonyms or []
        retains = retains or []

        columns = [col for col in self.columns.values if col not in deletes]
        # nodes for bayesian networks, which does not include pseudonym columns
        # or non-categorical string columns.
        nodes = set()
        for col in columns:
            if col in pseudonyms or (
                    self[col].type == 'string' and not self[col].categorical):
                continue
            nodes.add(col)
        # main steps of private bayesian network for synthesis
        # encode dataset into bin indexes for bayesian network
        indexes = DataFrame()
        for col in nodes:
            indexes[col] = self[col].bin_indexes()
        if indexes.shape[1] < 2:
            print('Warning: when there is only one attribute in dataset, this '
                  'algorithm inject noises to its probability distribution.')
            _cols = list(indexes)
            prs = noisy_distributions(indexes, _cols, epsilon)
            probability = {_cols[0]: normalize_distribution(prs['freq']).tolist()}
            return None, probability

        # Bayesian network is defined as a set of AP (attribute-parent) pairs.
        # e.g. [(x1, p1), (x2, p2), ...], and pi is the parents of xi.
        #
        # The algorithm follows the composability property of differential
        # privacy, so the privacy budget is split to two parts.
        network = greedy_bayes(indexes, epsilon / 2, degree=degree,
                               retains=retains)
        cond_prs = noisy_conditionals(network, indexes, epsilon / 2)
        return network, cond_prs

    def to_pattern(self, epsilon=0.1, degree=2, pseudonyms=None,
                   deletes=None, retains=None) -> dict:
        """
        Serialize this dataset's patterns into a dict object.
        """
        network, cond_prs = self._construct_bayesian_network(
            epsilon, degree=degree, pseudonyms=pseudonyms, deletes=deletes,
            retains=retains)
        pattern = dict({
            "attrs": {col: self[col].to_pattern() for col in self.columns
                      if col not in (deletes or [])},
            "config": {"pseudonyms": pseudonyms},
            "network": network,
            "prs": cond_prs,
            "records": self._records
        })
        return pattern

    def to_pattern_file(self, path, epsilon=0.1, degree=2, pseudonyms=None,
                        deletes=None, retains=None) -> None:
        """
        Serialize this dataset's patterns into a json file.
        """
        pattern = self.to_pattern(epsilon, degree, pseudonyms, deletes, retains)
        with open(path, 'w') as fp:
            json.dump(pattern, fp, indent=2)

    def synthesize(self, epsilon=0.1, degree=2,
                   pseudonyms=None, deletes=None, retains=None, records=None):
        """
        Synthesize data set by a bayesian network to infer attributes'
        dependence relationship and differential privacy to keep differentially
        private.
        """
        deletes = deletes or []
        pseudonyms = pseudonyms or (
                self._config is not None and self._config['pseudonyms']) or []
        retains = retains or []
        records = records if records is not None else self._records

        if self._network is None and self._cond_prs is None:
            self._network, self._cond_prs = self._construct_bayesian_network(
                epsilon, degree=degree, pseudonyms=pseudonyms, deletes=deletes,
                retains=retains)

        # if bayesian network is None, and probability is not None, that means
        # there is only one column in the dataset.
        if self._network is None and self._cond_prs is not None:
            # this is the only column label in this dataset
            col = self.columns[0]
            prs = self._cond_prs[col]
            from numpy import random
            sampling = Series(random.choice(len(prs), size=records, p=prs))
            attr = self[col].choice(indexes=sampling)
            return DataFrame(attr, columns=self.columns)

        columns = [col for col in self.columns.values if col not in deletes]
        sampling = self._sampling_dataset(self._network, self._cond_prs, records)
        frame = DataFrame(columns=columns)
        for col in self.columns:
            attr = self[col]
            if col in deletes:
                continue
            if col in pseudonyms:  # pseudonym column is not in bayesian network
                frame[col] = attr.pseudonymize(size=records)
                continue
            if col in retains:
                frame[col] = attr.retain(records)
                continue
            if col in sampling:
                frame[col] = attr.choice(indexes=sampling[col])
                continue
            if not attr.categorical:
                frame[col] = attr.random()
            else:
                frame[col] = attr.choice()
        return frame
