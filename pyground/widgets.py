import ipywidgets as widgets
from IPython.display import display, Markdown
from pathlib import Path
from pyground.file_utils import file_exists
from typing import List


def multiple_select_filenames(names: List[str], path: str, mask: str = "{}"):
    """
    Displays checkboxes for those filenames built from the path + mask([names])
    These names correspond to filenames, and they are marked if they don't exist.

    Example
        >>> names = ['sachs', 'sachs_long', 'toy', 'insurance']
        >>> path = '/tmp'
        >>> checkboxes = multiple_select_filenames(names, path, "pref_{}.csv")
        ...
        >>> to_reprocess = items_selected(checkboxes)

    Args:
        names (list): list of names
        path (str): The path where looking for the filenames
        mask (str): The mask to be used to build the filenames. This is usually a
            prefix, suffix or something more elaborated.

    Returns:
        widgets.Checkbox

    """
    existing_filenames = dict()
    for label in names:
        filename = mask.format(label)
        filename = str(Path(path, filename))
        if file_exists(filename, path):
            existing_filenames[label] = filename
        else:
            existing_filenames[label] = None

    display(Markdown('What models to fit?<br>Those selected will be loaded from disk.'))
    checkboxes = [
        widgets.Checkbox(value=existing_filenames[label] == None, description=label)
        for label in names]
    output = widgets.VBox(children=checkboxes)
    display(output)
    return checkboxes


def items_selected(selection: widgets.Checkbox):
    """ Returns what values have been selected from checkboxes list (selection) """
    selected_data = []
    for i in range(0, len(selection)):
        if selection[i].value == True:
            selected_data = selected_data + [selection[i].description]

    return selected_data
