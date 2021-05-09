# os.environ['PYTHONHASHSEED'] = '0'

import random as rand
import numpy as np
import pandas as pd

from prettytable import PrettyTable
from random import randint, random
from scipy.special import expit  # this is the sigmoid function
from scipy.stats import norm, lognorm
from sklearn.preprocessing import RobustScaler
from typing import List


def letter_in_string(string, letter):
    """Given a string, determine if that string contains the letter.
    Parameters:
      - string: a sequence of letters
      - letter: a string character.

    Return values:
      - index position if found,
      - -1 if ValueError exception is raised.
    """
    try:
        return string.index(letter)
    except ValueError:
        return -1


def which_string(strings, letter, group_index=0):
    """Return the string and position within that string where the letter
    passed as argument is found.

    Arguments:
      - strings: an array of strings.
      - letter: a single character to be found in strings

    Return values:
      - Tuple with index of string containing the letter, and position within
        the string. In case the letter is not found, both values are -1.
    """
    if len(strings) == 0:
        return -1, -1
    pos = letter_in_string(strings[0], letter)
    if pos != -1:
        return group_index, pos
    else:
        return which_string(strings[1:], letter, group_index + 1)


def previous(objects_array: object, pos: int):
    """
    Return the object at pos - 1 position, only if pos != 0, otherwise
    returns None
    """
    if pos == 0:
        return None
    else:
        return objects_array[pos - 1]


def print_progbar(percent: float, max: int = 20, do_print=True,
                  **kwargs: str) -> str:
    """ Prints a progress bar of max characters, with progress up to
    the passed percentage

    :param percent: the percentage of the progress bar completed
    :param max: the max width of the progress bar
    :param do_print: print the progbar or not
    :param **kwargs: optional arguments to the `print` method.

    Example
    -------

    >>> print_progbar(0.65)
    >>> "[=============·······]"

    """
    done = int(np.ceil(percent * 20))
    remain = max - done
    pb = "[" + "=" * done + "·" * remain + "]"
    if do_print is True:
        print(pb, sep="", **kwargs)
    else:
        return pb


def reset_seeds(my_seed=1):
    """
    Reset all internal seeds to same value always
    """
    np.random.seed(my_seed)
    rand.seed(my_seed)


def dict2table(dictionary: dict) -> str:
    """
    Converts a table into an ascii table.
    """
    t = PrettyTable()
    t.field_names = ['Parameter', 'Value']
    for header in t.field_names:
        t.align[header] = 'l'

    def tabulate_dictionary(t: PrettyTable, d: dict,
                            name: str = None) -> PrettyTable:
        for item in d.items():
            if isinstance(item[1], dict):
                t = tabulate_dictionary(t, item[1], item[0])
                continue
            sep = '.' if name is not None else ''
            prefix = '' if name is None else name
            t.add_row([f"{prefix}{sep}{item[0]}", item[1]])
        return t

    return str(tabulate_dictionary(t, dictionary))


def random_coefficients(degree: int, prob_zero=0.2) -> List:
    """
    Generate a list of random coefficients of the degree passed

    Arguments:
        - degree: The degree of the polynomial
        - prob_zero: The probability of zeroing a coefficient.

    Returns:
        A list with coefficients for each degree.

    Examples:
        >>> polynomial = random_coefficients(3)
        >>> print(polynomial)
        >>> [ +2.22  -0.00  -3.13 ]

    """
    poly = []
    for i, d in enumerate(range(degree)):
        beta = random() * randint(-10, 10)
        zero_coeff = 0.0 if random() < prob_zero else 1.0
        poly.append(beta * zero_coeff)
    return poly


def evaluate_poly(poly: List, x: float) -> float:
    """
    Evaluates a polynomial of certain degree from the list of coefficientes

    Arguments:
        - poly: A list with the coefficients of the polynomial
        - x: the value to be pluged in onto the polynomial
    Returns:
        A float value corresponding to the evaluation of the poly on "x"

    Examples:
        >>> polynomial = random_coefficients(3)
        >>> print(polynomial)
        >>> [ -0.82  +7.96  -3.83 ]
        >>> evaluate_poly(polynomial, x=1.0)
        >>> 3.3088
    """
    v = 0.0
    for degree, beta in enumerate(poly):
        v += beta * np.power(x, degree)
    return v


def add_extra_features(num_feats: int,
                       k: np.array,
                       degree=3) -> np.ndarray:
    """
    Generates features dependent on the series "k", based on a random polynomial

    Arguments:
    ----------
        - num_feats: the number of features to generate
        - k: the feature on which the new ones will depend on.
        - degree: The degree of each random polynomial

    Returns:
    --------
        A matrix with k.shape[0] x num_feats new values

    Examples:
    ---------
        >>> X = np.random.rand(3, 3)
        >>> ef = add_extra_features(1, X[:, -1])
        >>> X = np.append(X, ef, axis=1)
        >>> print(X)
            [[ 0.23941487  0.1282286   0.78841238  0.34497778]
            [ 0.19726776  0.86035895  0.31766776 -0.49607576]
            [ 0.83557953  0.71509514  0.46714884 -0.28545388]]
    """
    extra_features = []
    for n in range(num_feats):
        poly = random_coefficients(degree)
        y = list(map(lambda x: evaluate_poly(poly, x), k))
        extra_features.append(y)
    return np.array(extra_features).T


def gen_toy_dataset(mu=0, sigma=1., s=0.25, sigma_z0=3.0, sigma_z1=5.,
                    num_samples=1000, num_feats=5, scale=False):
    """
    Generate a toy dataset with 5 variables (or more), and the following causal
    relationship among them: z->x, z->t, z->y, t->y, k1, k2->k1, k3->k1

    Arguments:
    ----------
        - mu: mean for distributions of indep variables
        - sigma: variance of distributions of indep variables
        - s: mean for the lognormal distr.
        - sigma_z0: parameter to compute "x". 
        - sigma_z1: parameter to compute "x".
        - num_samples: Number of samples to generate
        - num_feats: number of features to generate. Min value is 5.
        - scale: Whether scaling the resulting DataFrame with RobustScaler
                 (default is False)

    Returns:
    --------
        A dataframe with 'num_feat' features following a given causal
        relationship, and the true structure of it.


    Examples:
    ---------
        >>> from pyground.utils import gen_toy_dataset
        >>> toy_dataset, true_order = gen_toy_dataset(num_samples=5)
        >>> toy_dataset
                    z         x         t         y         k
            0 -0.165956 -1.603104  1.144675  2.854501  3.007446
            1  0.440649 -1.060788  3.766573 -0.978671  4.682616
            2 -0.999771  0.381785  2.226256  0.372360 -1.865758
            3 -0.395335 -0.256640  4.783731 -5.642051  1.923226
            4 -0.706488  0.654393  0.526440 -2.708605  3.763892
    """
    reset_seeds()

    def fx(z):
        return (sigma_z1 * sigma_z1 * z) + (sigma_z0 * sigma_z0 * (1 - z))

    def ft(z):
        return 0.75 * z + 0.25 * (1 - z)

    def fy(T):
        return (expit(3. * (T[0] + 2. * (2. * T[1] - 1.))))

    z = lognorm.rvs(s=0.25, scale=1.0, size=num_samples)
    x_z = [norm.rvs(loc=zi, scale=fx(zi)) for zi in z]
    t_z = np.array(list(map(ft, z)))
    y_t_z = np.array(list(map(fy, zip(z, t_z))))
    k = norm.rvs(loc=0.0, scale=1.0, size=num_samples)
    features = np.array([x_z, t_z, y_t_z, z, k])
    column_names = ['x', 't', 'y', 'z', 'k']
    true_structure = {'z': ['x', 'y', 't'], 't': ['y']}

    # Add extra features if needed
    num_extra_features = num_feats - 5
    if num_extra_features > 0:
        extra_features = add_extra_features(num_feats - 5, k)
        features = np.append(features, extra_features, axis=1)
        true_structure['k'] = []
        # set the name of the extra columns (k1, k2, k3...)
        for i in range(num_extra_features):
            column_names.append(f'k{i}')
            true_structure['k'].append(f'k{i}')

    # Transform everything in a dataframe
    dataset = pd.DataFrame(data=features, columns=column_names)
    dataset = dataset.astype(np.float64)
    if scale is True:
        scaler = RobustScaler()
        dataset = pd.DataFrame(data=scaler.fit_transform(dataset),
                               columns=['x', 't', 'y', 'z', 'k'])

    return dataset, true_structure
