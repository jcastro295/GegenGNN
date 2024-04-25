'''

`checkpoint.py`
---------------

    This file contains the functions to save and load a model checkpoint

'''

import torch
import torch.nn as nn

from src.utils.logger import set_logger
from src.utils.printing import color_text

logger = set_logger('checkpoint')


def save_checkpoint(filename:str, save_dict:dict):
    '''
    Save a model checkpoint

    Parameters
    ----------
    filename: str
        The name of the checkpoint file
    save_dict: dict
        The dictionary containing the model information

    Returns
    -------
    None
    '''

    # first things first, we need to make a dictionary of the different pieces
    logger.debug(color_text('Constructing a dictionary with model information', color='gray', style='bold'))

    torch.save(save_dict, filename)

    logger.debug(color_text(f'Checkpoint file saved at {filename}', color='green', style='bold'))


def load_checkpoint(file_path:str):
    '''

    Load a model checkpoint

    Parameters
    ----------
    file_path: str
        The path to the checkpoint file

    Returns
    -------
    None
    '''

    logger.debug(color_text('A checkpoint file was found, attempting to load', color='gray', style='bold'))

    # load in the checkpoint file and map to the cpu so that we don't run into errors
    loaded_dict = torch.load(file_path)

    logger.debug(color_text('Successfully loaded the checkpoint file', color='green', style='bold'))

    return loaded_dict


def load_single_checkpoint(
                file_path:str,
                model:nn.Module,
                optimizer:nn.Module,
                scheduler:nn.Module,
                device='cpu'
                ):
    '''

    Load a model checkpoint

    Parameters
    ----------
    file_path: str
        The path to the checkpoint file
    model: nn.Module
        The model to load
    optimizer: nn.Module
        The optimizer to load
    scheduler: nn.Module
        The scheduler to load
    train_loss: list
        The training loss to load
    val_loss: list
        The validation loss to load
    device: str
        The device to load the model to

    Returns
    -------
    None
    '''

    logger.debug(color_text('A checkpoint file was found, attempting to load', color='gray', style='bold'))

    # load in the checkpoint file and map to the cpu so that we don't run into errors
    loaded_dict = torch.load(file_path)

    logger.debug(color_text('Successfully loaded the checkpoint file', color='green', style='bold'))

    # we then have the epoch
    logger.debug(color_text('Pulling out the epoch', color='gray', style='bold'))
    epoch = loaded_dict['epoch']
    logger.debug(color_text(f'Previous epoch was {epoch}', color='gray', style='bold'))

    # load the model state dict
    logger.debug(color_text('Loading the model state dictionary', color='gray', style='bold'))
    model.load_state_dict(loaded_dict['state_dict'])

    # load the optimizer state dict
    logger.debug(color_text('Loading the optimizer state dictionary', color='gray', style='bold'))
    optimizer.load_state_dict(loaded_dict['optimizer'])

    # the adam optimizer needs to step through the states and move it to the gpu
    logger.debug(color_text(f'Moving optimizer state values to {device}', color='gray', style='bold'))
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # load the scheduler state dict
    logger.debug(color_text('Loading the scheduler state dictionary', color='gray', style='bold'))
    scheduler.load_state_dict(loaded_dict['scheduler'])

    logger.debug(color_text('Loading train loss', color='gray', style='bold'))
    train_loss = loaded_dict['train_loss']

    logger.debug(color_text('Loading validation loss', color='gray', style='bold'))
    val_loss = loaded_dict['val_loss']

    logger.debug(color_text('Checkpoint loaded successfully', color='green', style='bold'))

    return epoch, model, optimizer, scheduler, train_loss, val_loss