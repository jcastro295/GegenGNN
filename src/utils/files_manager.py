"""

`files_manager.py`
------------------

    This module contains functions for managing files and directories.

"""

import os
import datetime
from pathlib import Path

import scipy.io as sio

from src.utils.logger import set_logger
from src.utils.printing import color_text

logger = set_logger('files_manager')

def get_directory_for_results(folder_path, new_name):
    """Creates a unique directory for saving results
    It first creates a path object and checks if it exists.
    If it doesn't exist, then it makes the directory and then returns the path
    object. If it does exist, then it starts adding _XX digits to the end
    of the folder name and increments by one every time a folder exists. Then,
    it makes the folder and returns the path.
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # let's append the date to the path
    new_name = datetime.datetime.now().strftime(
        "%Y_%m_%d_") + new_name

    # so, we need to get a save folder built up for this run
    save_folder = Path(
        os.path.join(os.path.expanduser(folder_path)), new_name)

    # if it's already a directory
    if save_folder.is_dir():
        # then we need to add a unique a number to the end
        logger.info(color_text('Save folder already found, making a new one!', color='gray', style='bold'))
        found_path = False
        i = 1
        while not found_path:
            save_folder = save_folder.parent / (new_name + f"_{i:02d}")
            if save_folder.is_dir():
                i += 1
            else:
                save_folder.mkdir()
                found_path = True
    else:
        save_folder.mkdir()

    return save_folder


def save_results(path:str, filename:str, data:dict):
    """
    Saves a dictionary of data to a file

    Parameters
    ----------
    path : str
        Path to save the file
    filename : str
        Name of the file
    data : dict
        Dictionary of data to save
    """

    save_path = os.path.join(path, filename)
    sio.savemat(save_path, data)

    logger.info(color_text(f'Saved results to {save_path}', color='green', style='bold'))