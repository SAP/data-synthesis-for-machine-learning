"""
BiFrame for data synthesis.
"""

import logging
import numpy as np
import pandas as pd

from pandas import Index
from sklearn.model_selection import train_test_split

from ds4ml.dataset import DataSet
from ds4ml.utils import train_and_predict, normalize_range
from ds4ml.metrics import jensen_shannon_divergence, relative_error

logger = logging.getLogger(__name__)


class BiFrame(object):
    def __init__(self, first: pd.DataFrame, second: pd.DataFrame,
                 categories=[]):
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
            logger.info(f"BiFrame works on columns: {cols}.")

        # left and right data set (ds)
        self.fst = DataSet(first[cols], categories=categories)
        self.snd = DataSet(second[cols], categories=categories)
        self._columns = sorted(cols)

        # Make sure that two dataset have same domain for categorical
        # attributes, and same min, max values for numerical attributes.
        for col in self._columns:
            # If current column is not categorical, will ignore it.
            if not self.fst[col].categorical or not self.snd[col].categorical:
                continue
            d1, d2 = self.fst[col].domain, self.snd[col].domain
            if not np.array_equal(d1, d2):
                if self.fst[col].categorical:
                    domain = np.unique(np.concatenate((d1, d2)))
                else:
                    domain = [min(d1[0], d2[0]), max(d1[1], d2[1])]
                self.fst[col].domain = domain
                self.snd[col].domain = domain

    @property
    def columns(self):
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
            counts1 = self.fst[column].counts(bins=bins)
            counts2 = self.snd[column].counts(bins=bins)
        else:
            min_, max_ = self.fst[column].domain
            # the domain from two data set are same;
            # extend the domain to human-readable range
            bins = normalize_range(min_, max_ + 1)
            counts1 = self.fst[column].counts(bins=bins)
            counts2 = self.snd[column].counts(bins=bins)
            # Note: index, value of np.histogram has different length
            bins = bins[:-1]
        # stack arrays vertically
        return bins, np.vstack((counts1, counts2))

    def describe(self):
        """
        Give descriptive difference between two data sets, which concluded
        relative errors, and jsd divergence.
        Return a panda.DataFrame, whose columns are two dataset's columns, and
        indexes are a array of metrics, e.g. ['err', 'jsd'].
        """
        df1 = self.err()
        df2 = self.jsd()
        return pd.concat([df1, df2])

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
        if (not self.fst[label].categorical or
                not self.snd[label].categorical):
            raise ValueError(f'Classifier can not run on non-categorical '
                             f'column: {label}')
        from sklearn.metrics import confusion_matrix

        def split_feature_label(df: pd.DataFrame):
            # TODO need improve sub_cols
            sub_cols = [attr for attr in df.columns if attr.startswith(label)]
            if len(sub_cols) == 0:
                return df, None
            is_one_class = len(sub_cols) == 2
            if is_one_class:
                # For one class, there are two sorted values.
                # e.g. ['Yes', 'No'] => [[0, 1],
                #                        [1, 0]]
                # Choose second column to represent this attribute.
                label_ = sub_cols[1]
                return df.drop(sub_cols, axis=1), df[label_]
            else:
                try:
                    # merge multiple columns into one column:
                    # [Name_A, Name_B, ..] => Name
                    _y = df[sub_cols].apply(lambda x: Index(x).get_loc(1),
                                            axis=1)
                    return df.drop(sub_cols, axis=1), _y
                except KeyError as e:
                    print(e)
                    print(sub_cols)
                    print(df[sub_cols])

        # If test dataset is not provided, then split 20% of original dataset
        # for testing.
        if test is None:
            fst_train, test = train_test_split(self.fst, test_size=0.2)
            snd_train, _ = train_test_split(self.snd, test_size=0.2)
        else:
            fst_train = self.fst
            snd_train = self.snd
        # ts = self.first.encode(data=fst_train)
        fst_train_x, fst_train_y = split_feature_label(
                                            self.fst.encode(data=fst_train))
        test_x, test_y = split_feature_label(self.fst.encode(data=test))
        snd_train_x, snd_train_y = split_feature_label(
                                            self.fst.encode(data=snd_train))

        # construct svm classifier, and predict on the same test dataset
        fst_predict_y = train_and_predict(fst_train_x, fst_train_y, test_x)
        snd_predict_y = train_and_predict(snd_train_x, snd_train_y, test_x)

        columns = self.fst[label].bins
        labels = range(len(columns))
        # If test dataset has the columns as class label for prediction, return
        # two expected scores: (self.first) original dataset's and (self.second)
        # anonymized dataset's confusion matrix.
        if label in test:
            fst_matrix = confusion_matrix(test_y, fst_predict_y, labels=labels)
            snd_matrix = confusion_matrix(test_y, snd_predict_y, labels=labels)
            # normalize the confusion matrix
            # fst_matrix = fst_matrix.astype('float') / fst_matrix.sum(axis=1)
            # snd_matrix = snd_matrix.astype('float') / snd_matrix.sum(axis=1)
            return (pd.DataFrame(fst_matrix, columns=columns, index=columns),
                    pd.DataFrame(snd_matrix, columns=columns, index=columns))
        # If test dataset does not have the class label for prediction, return
        # their predicted values.
        else:
            matrix = confusion_matrix(fst_predict_y, snd_predict_y,
                                      labels=labels)
            return pd.DataFrame(matrix, columns=columns, index=columns)

    def to_html(self, buf=None, title='Evaluation Report', info=True,
                distribute=True, correlate=True, classifier=None, labels=None,
                test=None):
        """
        Render the evaluation result of two data set as an HTML file.

        Parameters
        ----------
        buf : optional
            buffer to write to

        title : str
            title of evaluation report

        info : bool, default true
            show basic information of two data set, including relative error,
            and Jensen-Shannon divergence (jsd).

        distribute : bool, default true
            show distribution of each attribute.

        correlate : bool, default true
            show correlation of pair-wise attributes.

        classifier : str
            use classifier to train data set on one or more columns (defined by
            parameter 'label') and show prediction result on the evaluation
            report. Optional classifier: SVM.

        labels : list of column names
            column name, or a list of column names separated by comma, used for
            classification task.

        test : pd.DataFrame
            test data for classification, and other machine learning tasks.
        """
        from ds4ml.utils import (plot_histogram, plot_heatmap,
                                 plot_confusion_matrix)
        from mako.template import Template
        import os
        old_cwd = os.getcwd()
        os.chdir(os.path.dirname(__file__))
        template = Template(filename='template/report.html')
        os.chdir(old_cwd)

        topics = []
        content = {}
        # format different kinds of evaluation result to unified style
        if info:
            topics.append('basic')
            content['basic'] = [self.describe().to_dict('split')]

        if distribute:
            topics.append('dist')
            content['dist'] = []
            for col in self.columns:
                bins, counts = self.dist(col)
                svg = plot_histogram(bins, counts)
                content['dist'].append({'name': col, 'columns': bins,
                                        'data': counts, 'path': svg})

        if correlate:
            topics.append('corr')
            content['corr'] = []
            source_mi, target_mi = self.corr()
            source_svg = plot_heatmap(source_mi)
            target_svg = plot_heatmap(target_mi)
            content['corr'].append({'matrix': source_mi.to_dict('split'),
                                    'path': source_svg})
            content['corr'].append({'matrix': target_mi.to_dict('split'),
                                    'path': target_svg})

        if labels is not None:
            topics.append('svm')
            content['svm'] = []
            for col in labels:
                in_test = (test is not None and col in test) or (test is None)
                if in_test:
                    # When class label in svm classify test data, try to match
                    # two predicted result with the actual data, and so, there
                    # will be two confusion matrix diagrams.
                    try:
                        source_cm, target_cm = self.classify(col, test=test)
                        vrange = (
                            min(source_cm.values.min(), target_cm.values.min()),
                            max(source_cm.values.max(), target_cm.values.max()))
                        path = (
                            plot_confusion_matrix(source_cm, vrange=vrange,
                                                  xlabel='raw',
                                                  ylabel='actual'),
                            plot_confusion_matrix(target_cm, vrange=vrange,
                                                  xlabel='synth',
                                                  ylabel='actual'))
                        content['svm'].append({'column': col, 'path': path})
                    except ValueError as e:
                        print(e)
                else:
                    # If not, will compare two predicted result.
                    try:
                        cm = self.classify(col, test=test)
                        # make path's type: 1-tuple
                        path = (plot_confusion_matrix(cm, xlabel='synth',
                                                      ylabel='raw'),)
                        content['svm'].append({'column': col, 'path': path})
                    except ValueError as e:
                        print(e)

        svms = content['svm'] if 'svm' in content else []
        if buf:
            with open(buf, 'w+', encoding='utf-8') as file:
                file.write(template.render(title=title,
                                           basics=content['basic'],
                                           dists=content['dist'],
                                           corrs=content['corr'],
                                           svms=svms))
