"""
BiFrame for data synthesis.
"""

import warnings
import numpy as np
import pandas as pd

from pandas import Index
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from ds4ml.dataset import DataSet
from ds4ml.utils import train_and_predict, normalize_range
from ds4ml.metrics import jensen_shannon_divergence, relative_error


def split_feature_class(label: str, frame: pd.DataFrame):
    """ Split features and class from encode dataset. `label` is the class, and
    `frame` is the dataset which has been encoded.
    """
    sub_cols = [attr for attr in frame.columns if attr.startswith(label)]
    if len(sub_cols) <= 1:
        return frame, None
    is_one_class = len(sub_cols) == 2
    if is_one_class:
        # For one class, there are two sorted values.
        # e.g. ['Yes', 'No'] => [[0, 1],
        #                        [1, 0]]
        # Choose second column to represent this attribute.
        label_ = sub_cols[1]
        return frame.drop(sub_cols, axis=1), frame[label_]

    # merge multiple columns into one column: [Name_A, Name_B, ..] => Name
    y = frame[sub_cols].apply(lambda x: Index(x).get_loc(1), axis=1)
    return frame.drop(sub_cols, axis=1), y


class BiFrame:
    def __init__(self, first: pd.DataFrame, second: pd.DataFrame,
                 categories=None):
        """
        BiFrame class that contains two data sets, which currently provides
        kinds of analysis methods from distribution, correlation, and some
        machine learning tasks.
        Especially, if the input data sets are source and synthesized dataset,
        this class can be used to evaluate the utility and privacy of
        synthesized data set.

        Parameters
        ----------
        first : {pandas.DataFrame}
            first data set (i.e. original dataset)

        second : {pandas.DataFrame}
            second data set (i.e. synthesized dataset)

        categories : list of columns
            Column names whose values are categorical.
        """
        # To compare two data sets, make sure that they have same columns. If
        # not, compare them on their common columns.
        cols = set(first.columns) & set(second.columns)
        if len(cols) != len(first.columns) or len(cols) != len(second.columns):
            warnings.warn(f"The evaluator works on the common columns: {cols}.")

        categories = [] if categories is None else categories
        self.fst = DataSet(first[cols], categories=categories)
        self.snd = DataSet(second[cols], categories=categories)
        self._columns = sorted(cols)

        # Make sure that two dataset have same domain for categorical
        # attributes, and same min, max values for numerical attributes.
        for col in self._columns:
            # If current column is not categorical, will ignore it.
            if not self.fst[col].categorical or not self.snd[col].categorical:
                continue
            fst_domain, snd_domain = self.fst[col].domain, self.snd[col].domain
            if not np.array_equal(fst_domain, snd_domain):
                if self.fst[col].categorical:
                    domain = np.unique(np.concatenate((fst_domain, snd_domain)))
                else:
                    domain = [min(fst_domain[0], snd_domain[0]),
                              max(fst_domain[1], snd_domain[1])]
                self.fst[col].domain = domain
                self.snd[col].domain = domain

    @property
    def columns(self):
        """ Return the common columns of two datasets. """
        return self._columns

    def err(self):
        """
        Return pairwise err (relative error) of columns' distribution.
        """
        # merge two frequency counts, and calculate relative difference
        frame = pd.DataFrame(columns=self.columns, index=['err'])
        frame.fillna(0)
        for col in self.columns:
            frame.at['err', col] = relative_error(self.fst[col].counts(),
                                                  self.snd[col].counts())
        return frame

    def jsd(self):
        """
        Return pairwise JSD (Jensen-Shannon divergence) of columns' distribution.
        """
        frame = pd.DataFrame(columns=self.columns, index=['jsd'])
        frame.fillna(0)
        for col in self.columns:
            frame.at['jsd', col] = jensen_shannon_divergence(
                self.fst[col].counts(), self.snd[col].counts())
        return frame

    def corr(self):
        """
        Return pairwise correlation and dependence measured by mi (mutual
        information).
        """
        return self.fst.mi(), self.snd.mi()

    def dist(self, column):
        """
        Return frequency distribution of one column.

        Parameters
        ----------
        column : str
            column name, whose distribution will be return
        """
        if column not in self.columns:
            raise ValueError(f"{column} is not in current dataset.")
        if self.fst[column].categorical:
            bins = self.fst[column].domain
            fst_counts = self.fst[column].counts(bins=bins)
            snd_counts = self.snd[column].counts(bins=bins)
        else:
            min_, max_ = self.fst[column].domain
            # the domain from two data set are same;
            # extend the domain to human-readable range
            bins = normalize_range(min_, max_ + 1)
            fst_counts = self.fst[column].counts(bins=bins)
            snd_counts = self.snd[column].counts(bins=bins)
            # Note: index, value of np.histogram has different length
            bins = bins[:-1]
        # stack arrays vertically
        return bins, np.vstack((fst_counts, snd_counts))

    def describe(self):
        """
        Give descriptive difference between two data sets, which concluded
        relative errors, and jsd divergence.
        Return a panda.DataFrame, whose columns are two dataset's columns, and
        indexes are a array of metrics, e.g. ['err', 'jsd'].
        """
        err_frame = self.err()
        jsd_frame = self.jsd()
        return pd.concat([err_frame, jsd_frame])

    def classify(self, label: str, test: pd.DataFrame = None):
        """
        Train two svm classifiers based on data sets, and predict class labels
        for test data. Return both error rates.

        Parameters
        ----------
        label : str
            classifier feature, key is one column in left data frame.
            It supports two-class and multi-class.

        test : {pandas.DataFrame}
            test frame, is test data for machine learning algorithms. If it is
            not provided, it will split 20% of left data frame as test data.

        Returns
        -------
        a DataFrame, e.g.
                         target                         source     target
                      male female                    male female male female
        source male   1    3        or actual male   1    3      1    2
               female 2    4                  female 2    4      3    4
        """
        if not self.fst[label].categorical or not self.snd[label].categorical:
            raise ValueError(f'Must classify on categorical column')

        # If test dataset is not provided, then split 20% of original dataset
        # for testing.
        if test is None:
            fst_train, test = train_test_split(self.fst, test_size=0.2)
            snd_train, _ = train_test_split(self.snd, test_size=0.2)
        else:
            fst_train = self.fst
            snd_train = self.snd

        fst_train_x, fst_train_y = split_feature_class(label, self.fst.encode(
            data=fst_train))
        snd_train_x, snd_train_y = split_feature_class(label, self.fst.encode(
            data=snd_train))
        test_x, test_y = split_feature_class(label, self.fst.encode(data=test))

        # construct svm classifier, and predict on the same test dataset
        fst_predict_y = train_and_predict(fst_train_x, fst_train_y, test_x)
        snd_predict_y = train_and_predict(snd_train_x, snd_train_y, test_x)

        columns = self.fst[label].bins
        labels = range(len(columns))
        # If test dataset has the columns as class label for prediction, return
        # two expected scores: (self.fst) original dataset's and (self.snd)
        # synthesized dataset's confusion matrix.
        if label in test:
            fst_matrix = confusion_matrix(test_y, fst_predict_y, labels=labels)
            snd_matrix = confusion_matrix(test_y, snd_predict_y, labels=labels)
            return (pd.DataFrame(fst_matrix, columns=columns, index=columns),
                    pd.DataFrame(snd_matrix, columns=columns, index=columns))

        # If test dataset does not have the class label for prediction, return
        # their predicted values.
        matrix = confusion_matrix(fst_predict_y, snd_predict_y, labels=labels)
        return pd.DataFrame(matrix, columns=columns, index=columns)

    def to_html(self, buffer, title='Evaluation Report', labels=None, test=None):
        """
        Render the evaluation result of two datasets to an HTML file.

        The result contains:
        + basic information of two data set (relative error, and Jensen-Shannon
            divergence (jsd));
        + distribution of each attribute;
        + correlation of pair-wise attributes;
        + classification result by SVM to train data set on one or more columns
            (defined by parameter 'labels' and 'test' dataset).

        Parameters
        ----------
        buffer
            buffer to write to

        title : str
            title of evaluation report

        labels : list of column names
            column name, or a list of column names separated by comma, used for
            classification task.

        test : pd.DataFrame
            test data for classification, and other machine learning tasks.
        """
        basics = [self.describe().to_dict('split')]
        svms = self._get_svm_classifier(labels=labels, test=test)

        template = BiFrame._construct_template()
        with open(buffer, 'w+', encoding='utf-8') as file:
            file.write(template.render(title=title, basics=basics,
                                       dists=self._get_dist(),
                                       corrs=self._get_corr(),
                                       svms=svms))

    def _get_svm_classifier(self, labels=None, test=None):
        if labels is None:
            return []

        from ds4ml.utils import plot_confusion_matrix
        svms = []
        for col in labels:
            in_test = (test is not None and col in test) or (test is None)
            if in_test:
                # When class label in svm classify test data, try to match
                # two predicted result with the actual data, and so, there
                # will be two confusion matrix diagrams.
                src_matrix, tgt_matrix = self.classify(col, test=test)
                vrange = (
                    min(src_matrix.values.min(), tgt_matrix.values.min()),
                    max(src_matrix.values.max(), tgt_matrix.values.max()))
                path = (
                    plot_confusion_matrix(src_matrix, vrange=vrange,
                                          xlabel='raw', ylabel='actual'),
                    plot_confusion_matrix(tgt_matrix, vrange=vrange,
                                          xlabel='synth', ylabel='actual'))
                svms.append({'column': col, 'path': path})
            else:
                # If not, will compare two predicted result.
                matrix = self.classify(col, test=test)
                # make path's type: 1-tuple
                path = (plot_confusion_matrix(matrix,
                                              xlabel='synth', ylabel='raw'))
                svms.append({'column': col, 'path': path})
        return svms

    @staticmethod
    def _construct_template():
        """ construct template from a html """
        from mako.template import Template
        import os
        old_cwd = os.getcwd()
        os.chdir(os.path.dirname(__file__))
        template = Template(filename='template/report.html')
        os.chdir(old_cwd)
        return template

    def _get_dist(self):
        """ return the distribution information """
        from ds4ml.utils import plot_histogram
        dists = []
        for col in self.columns:
            bins, counts = self.dist(col)
            svg = plot_histogram(bins, counts)
            dists.append({'name': col, 'columns': bins, 'data': counts,
                          'path': svg})
        return dists

    def _get_corr(self):
        """ return the pair-wise correlation """
        from ds4ml.utils import plot_heatmap
        corrs = []
        fst_mi, snd_mi = self.corr()
        fst_svg = plot_heatmap(fst_mi)
        snd_svg = plot_heatmap(snd_mi)
        corrs.append({'matrix': fst_mi.to_dict('split'), 'path': fst_svg})
        corrs.append({'matrix': snd_mi.to_dict('split'), 'path': snd_svg})
        return corrs
