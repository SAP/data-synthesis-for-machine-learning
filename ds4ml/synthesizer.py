# encoding: utf-8
"""
Algorithms to synthesize data set: currently differential privacy
"""
import warnings
import numpy as np

from itertools import product
from pandas import DataFrame, merge
from scipy.optimize import fsolve

from ds4ml.utils import mutual_information, normalize_distribution


# -----------------------------------------------------------------------------
# Algorithms PrivBayes: Private Data Release via Bayesian Networks


def calculate_sensitivity(n_rows, child, parents, binaries):
    """
    Lemma 4.1 Page 12: Sensitivity function for Bayesian network construction.

    Parameters
    ----------
    n_rows : int
        Number of tuples in dataset
    child : str
        One column name
    parents : tuple
        Parents of child, there may be multiple parents
    binaries : list
        List of binary columns
    """
    if child in binaries or (len(parents) == 1 and parents[0] in binaries):
        a = np.log(n_rows) / n_rows
        b = (1 / n_rows - 1) * np.log(1 - 1 / n_rows)
    else:
        a = (2 / n_rows) * np.log((n_rows + 1) / 2)
        b = (1 - 1 / n_rows) * np.log(1 + 2 / (n_rows - 1))
    return a + b


def usefulness(degree, n_rows, n_cols, epsilon, threshold):
    """
    Lemma 4.8 Page 16: Usefulness measure of each noisy marginal distribution
    """
    theta = n_rows * epsilon / ((n_cols - degree) * (2 ** (degree + 2)))
    return theta - threshold


def calculate_degree(n_rows, n_cols, epsilon):
    """
    Lemma 5.8 Page 16: The largest degree that guarantee theta-usefulness.
    """
    threshold = 5  # threshold of usefulness from Page 17
    default = min(3, int(n_cols / 2))
    args = (n_rows, n_cols, epsilon, threshold)
    warnings.filterwarnings("error")
    try:
        degree = fsolve(usefulness, np.array(int(n_cols / 2)), args=args)[0]
        degree = int(np.ceil(degree))
    except RuntimeWarning:
        warnings.warn('Degree of bayesian network is not properly computed!')
        degree = default
    if degree < 1 or degree > n_cols:
        degree = default
    return degree


def candidate_pairs(paras):
    """
    Return attribute-parents pairs, and their mutual information.
    """
    from itertools import combinations
    child, columns, n_parents, index, dataset = paras
    aps = []
    mis = []

    if index + n_parents - 1 < len(columns):
        for parents in combinations(columns[index + 1:], n_parents - 1):
            parents = list(parents)
            parents.append(columns[index])
            aps.append((child, parents))
            # TODO duplicate calculation of mutual information
            mi = mutual_information(dataset[child], dataset[parents])
            mis.append(mi)
    return aps, mis


def greedy_bayes(dataset: DataFrame, epsilon, degree=None, retains=None):
    """
    Algorithm 4, Page 20: Construct bayesian network by greedy algorithm.

    Parameters
    ----------
    dataset : DataFrame
        Encoded dataset
    epsilon : float
        Parameter of differential privacy
    degree : int
        Degree of bayesian network. If null, calculate it automatically.
    retains : list
        The columns to retain
    """
    dataset = dataset.astype(str, copy=False)
    n_rows, n_cols = dataset.shape
    retains = retains or []
    if not degree:
        degree = calculate_degree(n_rows, n_cols, epsilon)

    # mapping from column name to is_binary, because sensitivity is different
    # for binary or non-binary column
    binaries = [col for col in dataset if dataset[col].unique().size <= 2]
    more_retains = False
    if len(retains) == 0:
        root_col = np.random.choice(dataset.columns)
    elif len(retains) == 1:
        root_col = retains[0]
    else:
        root_col = np.random.choice(retains)
        more_retains = True

    # columns: a set that contains all attributes whose parent sets has been set
    columns = [root_col]
    if more_retains:
        left_cols = set(retains)
    else:
        left_cols = set(dataset.columns)
    left_cols.remove(root_col)
    network = []
    while len(left_cols) > 0:
        # ap: attribute-parent (AP) pair is a tuple. It is a node in bayesian
        # network, e.g. ('education', ['relationship']), there may be multiple
        # parents, depends on k (degree of bayesian network).
        aps = []
        # mi: mutual information (MI) of two features
        mis = []
        n_parents = min(len(columns), degree)
        # calculate the candidate set of attribute-parent pair
        tasks = [(child, columns, n_parents, index, dataset) for child, index in
                 product(left_cols, range(len(columns) - n_parents + 1))]
        # TODO: should use thread pool for large data set?
        candidates = list(map(candidate_pairs, tasks))
        for ap, mi in candidates:
            aps += ap
            mis += mi
        # find next child node in bayesian networks according to the biggest
        # mutual information or exponential mechanism
        if epsilon:
            index = sampling_pair(mis, aps, binaries, n_rows, n_cols, epsilon)
        else:
            index = mis.index(max(mis))
        network.append(aps[index])
        next_col = aps[index][0]
        columns.append(next_col)
        left_cols.remove(next_col)
        if len(left_cols) == 0 and more_retains:
            left_cols = set(dataset.columns) - set(retains)
            more_retains = False
    return network


def sampling_pair(mis, aps, binaries, n_rows, n_cols, epsilon):
    """
    Page 6 and 12: Sampling an attribute-parent pair from candidates by
    exponential mechanism.
    """
    deltas = []
    for child, parents in aps:
        sensitivity = calculate_sensitivity(n_rows, child, parents, binaries)
        delta = (n_cols - 1) * sensitivity / epsilon
        deltas.append(delta)
    prs = np.array(mis) / (2 * np.array(deltas))
    prs = np.exp(prs)
    prs = normalize_distribution(prs)
    return np.random.choice(list(range(len(mis))), p=prs)


def noisy_distributions(dataset, columns, epsilon):
    """
    Generate differentially private distribution by adding Laplace noise
    Algorithm 1 Page 9: parameters (scale, size) of Laplace distribution
    """
    data = dataset.copy()[columns]
    data['freq'] = 1
    freq = data.groupby(columns).sum()
    freq.reset_index(inplace=True)

    iters = [range(int(dataset[col].max()) + 1) for col in columns]
    domain = DataFrame(columns=columns, data=list(product(*iters)))
    # freq: the complete probability distribution
    freq = merge(domain, freq, how='left')
    freq.fillna(0, inplace=True)

    n_rows, n_cols = dataset.shape
    scale = 2 * (n_cols - (len(columns) - 1)) / (n_rows * epsilon)
    if epsilon:
        noises = np.random.laplace(0, scale=scale, size=freq.shape[0])
        freq['freq'] += noises
        freq.loc[freq['freq'] < 0, 'freq'] = 0
    return freq


def noisy_conditionals(network, dataset, epsilon):
    """
    Algorithm 1, Page 9: noisy conditional distribution probability
    """
    cond_prs = {}  # conditional probability distributions

    # distribution of one or more root node(s) in bayesian network
    root = network[0][1][0]
    # attributes [1, k]
    kattr = [root]
    for child, _ in network[:len(network[-1][1])]:
        kattr.append(child)

    kfreq = noisy_distributions(dataset, kattr, epsilon)
    root_prs = kfreq[[root, 'freq']].groupby(root).sum()['freq']
    cond_prs[root] = normalize_distribution(root_prs).tolist()

    # distributions of other child node(s) in bayesian network
    net_idx = 0
    for child, parents in network:
        cond_prs[child] = {}
        if net_idx < len(network[-1][1]):
            freq = kfreq.copy().loc[:, parents + [child, 'freq']]
        else:
            freq = noisy_distributions(dataset, parents + [child], epsilon)
        freq = DataFrame(freq[parents + [child, 'freq']]
                         .groupby(parents + [child]).sum())
        if len(parents) == 1:
            for parent in freq.index.levels[0]:
                prs = normalize_distribution(freq.loc[parent]['freq']).tolist()
                cond_prs[child][str([parent])] = prs
        else:
            for parent in product(*freq.index.levels[:-1]):
                prs = normalize_distribution(freq.loc[parent]['freq']).tolist()
                cond_prs[child][str(list(parent))] = prs
        net_idx = net_idx + 1
    return cond_prs
