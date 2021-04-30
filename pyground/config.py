"""
This class reads params from a YAML file and creates an object that
contains attributes named as the params in the file, accessible through
getters:

  object.parameter

in addition to classical dictionary access method

  object[parameter]

The structure of attributes is built recursively if they contain a dictionary.

  object.attr1.attr2.attr3

"""
from collections import defaultdict
from os import getcwd
from pathlib import Path

from yaml import safe_load, YAMLError

from pyground.logger import Logger
from pyground.utils import dict2table

debug = False


class Configuration(defaultdict):

    def __init__(self):
        # https://stackoverflow.com/a/45411093/892904
        super(Configuration, self).__init__()

    def __getattr__(self, key):
        """
        Check out https://stackoverflow.com/a/42272450
        """
        if key in self:
            return self.get(key)
        raise AttributeError('Key <{}> not present in dictionary'.format(key))

    def __setattr__(self, key, value):
        self[key] = value

    def __str__(self):
        return dict2table(self)

    @staticmethod
    def logdebug(*args, **kwargs):
        if debug is True:
            print(*args, **kwargs)

    def add_dict(self, this_object, param_dictionary, add_underscore=True):
        for param_name in param_dictionary.keys():
            self.logdebug('ATTR: <{}> type is {}'.format(
                param_name, type(param_dictionary[param_name])))

            if add_underscore is True:
                attribute_name = '{}'.format(param_name)
            else:
                attribute_name = param_name

            if type(param_dictionary[param_name]) is not dict:
                self.logdebug(' - Setting attr name {} to {}'.format(
                    attribute_name, param_dictionary[param_name]))
                setattr(this_object, attribute_name,
                        param_dictionary[param_name])
            else:
                self.logdebug(' x Dictionary Found!')

                self.logdebug('   > Creating new dict() with name <{}>'.format(
                    attribute_name))
                setattr(this_object, attribute_name, Configuration())

                self.logdebug('     > New Attribute <{}> type is: {}'.format(
                    attribute_name, type(getattr(this_object, attribute_name))
                ))
                new_object = getattr(this_object, attribute_name)

                self.logdebug('   > Calling recursively with dict')
                self.logdebug('     {}'.format(param_dictionary[param_name]))
                this_object.add_dict(new_object, param_dictionary[param_name])


def config(params_filename='params.yaml') -> Configuration:
    """
    Read the parameters from a filename

    Args:
    - params_filename: the name of the YAML file you want to use as source
        of parameters

    Returns:
    A customdict object containing the parameters read from file.
    """
    config = Configuration()
    cwd = Path(getcwd())
    params_path: str = str(cwd.joinpath(params_filename))

    try:
        with open(params_path, 'r') as stream:
            try:
                params_read = safe_load(stream)
                config.add_dict(config, params_read)
            except YAMLError as exc:
                print("YAML bad formatted. Ignored.")
                pass
    except FileNotFoundError as fnfe:
        print("No params.yaml parameters file. Taking defaults.")

    #
    # Set log_level and start the logger
    #
    if 'log_level' not in config:
        config.log_level = 3  # default value = WARNING
    config.log = Logger(config.log_level)

    return config
