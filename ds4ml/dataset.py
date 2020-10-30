"""
DataSet: data structure for potentially mixed-type Attribute.
"""

from pandas import DataFrame

from ds4ml.attribute import Attribute


class DataSet(DataFrame):

    _metadata = ['_categories']

    def __init__(self, data=None, index=None, columns=None, dtype=None,
                 copy=False, categories=None):
        """
        A DataFrame with categories information.

        Parameters
        ----------
        categories : list of columns
            Column names whose values are categorical.
        """
        DataFrame.__init__(self, data=data, index=index, columns=columns,
                           dtype=dtype, copy=copy)
        self.separator = '_'
        self._categories = categories or []

    @property
    def _constructor(self):
        return DataSet

    @property
    def _constructor_sliced(self):
        return Attribute

    # workaround: override method to add parameter 'categorical' for Attribute
    # constructor
    def _box_col_values(self, values, items):
        """
        Provide boxed values for a column.
        """
        klass = self._constructor_sliced
        return klass(values, index=self.index, name=items, fastpath=True,
                     categorical=items in self._categories)

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
        for col, attr in self.items():
            if data is not None and col not in data:
                continue
            if attr.categorical:
                subs = attr.encode(None if data is None else data[col])
                for label in attr.bins:
                    frame[col + self.separator + str(label)] = subs[label]
            else:
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

    def synthesize(self, epsilon=0.1, degree=2,
                   pseudonyms=None, deletes=None, retains=None, records=None):
        """
        Synthesize data set by a bayesian network to infer attributes'
        dependence relationship and differential privacy to keep differentially
        private.
        """
        deletes = deletes or []
        pseudonyms = pseudonyms or []
        retains = retains or []

        columns = [col for col in self.columns.values if col not in deletes]
        nodes = set()  # nodes for bayesian networks
        for col in columns:
            if col in pseudonyms or (
                    self[col].atype == 'string' and not self[col].categorical):
                continue
            nodes.add(col)
        # main steps of private bayesian network for synthesis
        # encode dataset into bin indexes for bayesian network
        indexes = DataFrame()
        for col in nodes:
            indexes[col] = self[col].bin_indexes()
        if indexes.shape[1] < 2:
            raise Exception('If infer bayesian network, it requires at least 2 '
                            'attributes in dataset.')

        # Bayesian network is defined as a set of AP (attribute-parent) pairs.
        # e.g. [(x1, p1), (x2, p2), ...], and pi is the parents of xi.
        #
        # The algorithm follows the composability property of differential
        # privacy, so the privacy budget is split to two parts.
        from ds4ml.synthesizer import greedy_bayes, noisy_conditionals
        network = greedy_bayes(indexes, epsilon / 2, degree=degree,
                               retains=retains)
        cond_prs = noisy_conditionals(network, indexes, epsilon / 2)

        records = records if records is not None else self.shape[0]
        sampling = self._sampling_dataset(network, cond_prs, records)
        frame = DataFrame(columns=columns)
        for col, attr in self.items():
            if col in deletes:
                continue
            if col in pseudonyms:
                frame[col] = attr.pseudonymize()
                continue
            if col in retains:
                frame[col] = attr
                continue
            if col in sampling:
                frame[col] = attr.choice(indexes=sampling[col])
            elif not attr.categorical:
                frame[col] = attr.random()
            else:
                frame[col] = attr.choice()
        return frame
