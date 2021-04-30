# os.environ['PYTHONHASHSEED'] = '0'

import random
from typing import List

import numpy as np
import pandas as pd

from pandas import DataFrame
from prettytable import PrettyTable
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


def gen_toy_dataset(N: int = 1000, scale=False) -> (DataFrame, dict):
    """
    Generate a toy dataset with 5 variables, and the following causal 
    relationship among them
    z->x, z->t, z->y, t->y, k

    Params
    ------
        - N: Number of samples to generate
        - scale: Whether scaling the resulting DataFrame with RobustScaler

    Example
    -------
        >>> from pyground.utils import gen_toy_dataset
        >>> toy_dataset, true_order = gen_toy_dataset(N=5)
        >>> toy_dataset
                    z         x         t         y         k
            0 -0.165956 -1.603104  1.144675  2.854501  3.007446
            1  0.440649 -1.060788  3.766573 -0.978671  4.682616
            2 -0.999771  0.381785  2.226256  0.372360 -1.865758
            3 -0.395335 -0.256640  4.783731 -5.642051  1.923226
            4 -0.706488  0.654393  0.526440 -2.708605  3.763892
    """
    reset_seeds()
    E1 = np.random.uniform(low=-1.0, high=1.0, size=N)
    E2 = np.random.uniform(low=-2.0, high=2.0, size=N)
    E3 = np.random.uniform(low=-3.0, high=3.0, size=N)
    E4 = np.random.uniform(low=-4.0, high=4.0, size=N)
    E5 = np.random.uniform(low=-5.0, high=5.0, size=N)

    df = DataFrame()
    df['z'] = E1
    df['x'] = np.power(df['z'], 2) + E2
    df['t'] = 4. * np.sqrt(np.abs(df['z'])) + E3
    df['y'] = 2. * np.sin(df['z']) + 2 * np.sin(df['t']) + E4
    df['k'] = E5

    if scale:
        scaler = RobustScaler()
        df[df.columns.values] = scaler.fit_transform(df[df.columns.values])

    true_structure = {'z': ['x', 'y', 't'], 't': ['y']}
    return df, true_structure
