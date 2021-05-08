# os.environ['PYTHONHASHSEED'] = '0'

import random
from typing import List

import numpy as np
import pandas as pd

from pandas import DataFrame
from prettytable import PrettyTable
from scipy.stats import bernoulli, norm, lognorm
from scipy.special import expit   # this is the sigmoid function
from sklearn.preprocessing import RobustScaler
from termcolor import colored


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


def reset_seeds(seed=1):
    """
    Reset all internal seeds to same value always
    """
    np.random.seed(seed)
    random.seed(seed)


def dict2table(dictionary: dict) -> str:
    """
    Converts a table into an ascii table.
    """
    t = PrettyTable()
    t.field_names = ['Parameter', 'Value']
    for header in t.field_names:
        t.align[header] = 'l'

    def tabulate_dictionary(t: PrettyTable, d: dict, name: str = None) -> PrettyTable:
        for item in d.items():
            if isinstance(item[1], dict):
                t = tabulate_dictionary(t, item[1], item[0])
                continue
            sep = '.' if name is not None else ''
            prefix = '' if name is None else name
            t.add_row([f"{prefix}{sep}{item[0]}", item[1]])
        return t

    return str(tabulate_dictionary(t, dictionary))


def gen_toy_dataset(mu=0, sigma=1., s=0.25, sigma_z0=3.0, sigma_z1=5.,
                    num_samples=1000, scale=False):
    """
    Generate a toy dataset with 5 variables, and the following causal 
    relationship among them
    z->x, z->t, z->y, t->y, k

    Params
    ------
        - mu: mean for distributions of indep variables
        - sigma: variance of distributions of indep variables
        - s: mean for the lognormal distr.
        - sigma_z0: parameter to compute "x". 
        - sigma_z1: parameter to compute "x".
        - num_samples: Number of samples to generate
        - scale: Whether scaling the resulting DataFrame with RobustScaler
                 (default is False)

    Example
    -------
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
        return (sigma_z1*sigma_z1*z) + (sigma_z0*sigma_z0*(1-z))

    def ft(z):
        return 0.75*z + 0.25*(1-z)

    def fy(T):
        return (expit(3.*(T[0] + 2.*(2.*T[1]-1.))))

    z = lognorm.rvs(s=0.25, scale=1.0, size=num_samples)
    x_z = [norm.rvs(loc=zi, scale=fx(zi)) for zi in z]
    t_z = np.array(list(map(ft, z)))
    y_t_z = np.array(list(map(fy, zip(z, t_z))))
    k = norm.rvs(loc=0.0, scale=1.0, size=num_samples)

    dataset = pd.DataFrame({'x': x_z, 't': t_z, 'y': y_t_z, 'z': z, 'k': k})
    dataset = dataset.astype(float)
    if scale is True:
        scaler = RobustScaler()
        dataset = pd.DataFrame(data=scaler.fit_transform(dataset),
                               columns=['x', 't', 'y', 'z', 'k'])

    true_structure = {'z': ['x', 'y', 't'], 't': ['y']}
    return dataset, true_structure
