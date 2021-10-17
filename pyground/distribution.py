import numpy as np
from matplotlib import pyplot as plt, gridspec
import seaborn as sns
import warnings

from scipy.stats import stats


def values_threshold(values, percentile=0.8, verbose=False):
    """
    Computes the value from which the accumulated sum of values represent
    the percentage passed as argument. The value is computed sorting the values in
    descending order, so the this metric determines what are the most important values.

    Parameters:
        - values (np.array): List of values (1D) to analyze
        - percentile (float): The percentage of the total sum of the values.

    Examples:
        >>>> a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>>> threshold, position = values_threshold(a, verbose=True)
        Values shape: 10
        Sum values: 55.00
        Sum@Percentile(0.80): 44.00
        Position @ percentile 0.80 in cum.sum: 6
        Threshold (under perc.0.80): 7.00
    """
    sum_values = values.sum()
    if verbose:
        print(f"Values shape: {values.shape[0]}")
        print(f"Sum values: {sum_values:.2f}")
        print(
            f"Sum@Percentile({percentile:.2f}): {sum_values * percentile:.2f}")
    cumsum = np.cumsum(sorted(values, reverse=True))
    # Substract because cumsum is reversed
    pos_q = values.shape[0] - np.where(cumsum < (sum_values * percentile))[
        0].max()
    if pos_q == values.shape[0]:
        pos_q -= 1
    if verbose:
        print(f"Position @ percentile {percentile:.2f} in cum.sum: {pos_q}")
    threshold = sorted(values)[pos_q]
    if verbose:
        print(f"Threshold (under perc.{percentile:.2f}): {threshold:.2f}")
    return threshold, pos_q


def plot_distribution(values: np.ndarray, percentile=None, **kwargs):
    """
    Plots histogram, density and values sorted. If percentile parameter is set,
    it is also plotted the position from which the values account for that
    percentage of the total sum.

    Parameters:
        - values (np.array): list of values (1D).
        - percentile (float): The percentage of the total sum of the values.
        - verbose(bool): Verbosity

    Return:
        - (threshold, position) (float, int): the value from which the cum
            sum accounts for the 'percentile' percentage of the total sum, and
            the position in the sorted list of values where that threshold is
            located.
    """
    verbose = kwargs.get('verbose', False)
    th, pos = None, None
    if percentile is not None:
        th, pos = values_threshold(values, verbose=verbose)

    fig = plt.figure(tight_layout=True, figsize=(8, 6))
    gs = gridspec.GridSpec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(values, edgecolor='white', alpha=0.5)
    ax1.set_title("Histogram of weights")

    ax2 = fig.add_subplot(gs[0, 1])
    sns.kdeplot(values, ax=ax2, bw_adjust=0.5)
    ax2.set_title("Density of weights")

    ax3 = fig.add_subplot(gs[1, :])
    x, y = range(len(values)), sorted(values)
    ax3.plot(x, y)
    ax3.set_title(f"{percentile * 100:.0f}% of total sum (>{th:.2f})")
    if percentile is not None:
        ax3.axvline(pos, linewidth=.5, c='red', linestyle='dashed')
        ax3.fill_between(x, min(y), y, where=x >= pos, alpha=0.2)

    fig.align_labels()
    plt.tight_layout()
    plt.show()

    return th, pos


def analyze_distribution(values, percentile=None, **kwargs):
    """
    Analyze the data to find what is the most suitable distribution type using
    Kolmogorov-Smirnov test.

    Arguments:
        values (np.array): List of values
        percentile (float): The percentile of cum sum down to which compute
            threshold.
        (Optional)
        plot (bool): Default is True
        verbose (bool): Default is False

    Return:
        Dictionary with keys: 'name' of the distribution, the
            'p_value' obtained in the test, the 'dist' itself as Callable, and
            the 'params' of the distribution. If parameter percentile is passed,
            the value from which the accumulated sum of values represent the
            percentage passed, under the key 'threshold'.
    """
    warnings.filterwarnings(action='ignore', category=RuntimeWarning)
    plot = kwargs.get('plot', True)
    verbose = kwargs.get('verbose', False)
    d = dict()

    if plot is True:
        d['threshold'], _ = plot_distribution(values, percentile,
                                              verbose=verbose)
        d['percentile'] = percentile

    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto",
                  "genextreme"]
    best_pvalue = 0.0
    for dist_name in dist_names:
        dist = getattr(stats, dist_name)
        param = dist.fit(values)

        # Applying the Kolmogorov-Smirnov test
        D, p = stats.kstest(values, dist_name, args=param)
        if p > best_pvalue:
            best_pvalue = p
            d['name'] = dist_name
            d['p_value'] = p
            d['dist'] = dist
            d['params'] = param

    return d
