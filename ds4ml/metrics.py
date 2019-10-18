"""
Metrics for data synthesis
"""
import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, confusion_matrix


def pairwise_mutual_information(frame: pd.DataFrame):
    """
    Return mutual information matrix for pairwise columns of a DataFrame.
    """
    columns = frame.columns.sort_values().to_list()
    mi = pd.DataFrame(columns=columns, index=columns, dtype=float)
    for row in columns:
        for col in columns:
            if pd.isnull(mi.at[row, col]):
                mi.at[row, col] = normalized_mutual_info_score(
                    frame[row].astype(str), frame[col].astype(str),
                    average_method='arithmetic')
            else:
                mi.at[row, col] = mi.at[col, row]
    return mi.round(3)


def jensen_shannon_divergence(p, q, base=2):
    """
    Return the Jensen-Shannon divergence between two 1-D arrays.

    Parameters
    ---------
    p : array
        left probability array
    q : array
        right probability array
    base : numeric, default 2
        logarithm base

    Returns
    -------
    jsd : float
        divergence of p and q
    """
    # If the sum of probability array p or q does not equal to 1, then normalize
    p = np.asarray(p)
    q = np.asarray(q)
    p = p / np.sum(p, axis=0)
    q = q / np.sum(q, axis=0)
    from scipy.spatial.distance import jensenshannon
    return round(jensenshannon(p, q, base=base), 4)


def error_rate(y_true, y_pred=None):
    """
    Return error (mis-classification) rate of one classifier result

    If there is only one parameter, it must be the confusion matrix;
    If there are two parameters, they must be true and predict labels;
    """
    if y_pred is None:
        if isinstance(y_true, pd.DataFrame):
            cm = y_true.values
        else:
            cm = y_true
    else:
        cm = confusion_matrix(y_true, y_pred)
    trace = np.trace(cm)
    sum = np.sum(cm)
    return round((sum - trace) / sum, 4)


def relative_error(x, y):
    """
    Return relative error of two variables: |x-y|/max(|x|, |y|)
    """
    m = np.maximum(np.abs(x), np.abs(y))
    return round(np.average(np.abs(x - y) / (m + 1e-6)), 4)
