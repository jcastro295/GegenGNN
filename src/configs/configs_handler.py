"""
configs_handler.py
-------------------

"""
from typing import Union
from collections.abc import MutableMapping
from dataclasses import dataclass, field

import toml


@dataclass
class Configs:
    """
    This handles the defaults for the GCN configs.

    A dataclass object was chosen because it can be initialized
    like a regular class without having to actually define the enormous
    amount of inputs that a single class would require.
    """

    # Basic configs
    run_name : str = ""
    run_description : str = ""
    output_folder : str = ""
    data_folder : str = ""
    dataset : str = ""
    dev_seed : int = 42
    val_seeds : list = field(default_factory=lambda: [42])
    test_seeds : list = field(default_factory=lambda: [42])
    no_cuda : bool = False
    gpu : int = 0

    # Preprocessing configs
    sampling_density : int = 0.1
    normalization : str = "none"
    layer_normalization : str = "neighbornorm"

    # Training configs
    model : str = "gegen"
    model_kwargs : dict = field(default_factory=dict)
    n_runs : int = None
    validation_size : float = 0.1
    epochs_to_save_checkpoint : int = 100
    save_checkpoint : bool = True
    early_stopping : bool = True
    patience : int = 100
    verbose : bool = True

    # hyperparameters options
    lr : Union[float, list] = 0.01
    epsilon : Union[float, list] = 0.05
    filter_order : Union[int, list] = 3
    alpha : Union[float, list] = 1
    heads : Union[int, list] = 1
    epochs : Union[int, list] = 1000
    dropout : Union[float, list] = 0.5
    k : Union[int, list] = 1
    gamma_norm : Union[float, list] = 1.0
    output : Union[int, list] = 1
    lambda_param : Union[float, list] = 4.82038e-4
    n_conv_layers : Union[int, list] = 1
    n_conv_hidden : Union[int, list] = 10
    n_linear_layers : Union[int, list] = 1
    n_linear_hidden : Union[int, list] = 10
    optim : str = "adam"
    weight_decay : float = 0.01
    optim_kwargs : dict = field(default_factory=dict)
    lr_scheduler : str = "none"
    lr_scheduler_kwargs : dict = field(default_factory=dict)
    criterion : str = "mse"
    criterion_kwargs : dict = field(default_factory=dict)


def extract_configs(d, depth=1, key='') -> dict:
    """
    Extracts the configs from a dictionary.

    Parameters
    ----------
    d : dict
        The dictionary to extract the configs from.
    depth : int, optional
        The depth to extract the configs from, by default 1.
    key : str, optional
        The key to use for the parent key, by default ''.

    Returns
    -------
    dict
        A dictionary with the extracted configs.
    """

    items = []
    for k, v in d.items():
        if isinstance(v, MutableMapping):
            items.extend(extract_configs(v, depth-1, key=k).items())
        else:
            if depth > 0:
                items.append((k, v))
            else:
                parent_key = key
                items.append((parent_key, d))

    return dict(items)


def toml_to_configs(toml_file:str) -> Configs:
    """
    Converts a toml file to a Configs object.

    Parameters
    ----------
    toml_file : str
        Path to the toml file.
    
    Returns
    -------
    configs : object
        A Configs object.
    """

    dump_file = toml.load(toml_file)

    configs = extract_configs(dump_file, depth=3)

    return Configs(**configs)
