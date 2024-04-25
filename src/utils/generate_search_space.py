import numpy as np

from src.configs.configs_handler import extract_configs
from .printing import color_text
from .logger import set_logger

logger = set_logger('search_space')

# Syntax: {'parameter' : 'mode'} -> 'mode' : 'uniform', 'discrete', 'fixed'
PARAMETER_DICT = {
        'alpha' : 'uniform',
        'heads' : 'discrete',
        'filter_order': 'discrete',
        'epsilon' : 'uniform',
        'dropout' : 'uniform',
        'k' : 'discrete',
        'gamma_norm' : 'uniform',
        'output' : 'discrete',
        'lr' : 'uniform',
        'n_conv_layers' : 'discrete',
        'n_conv_hidden' : 'discrete',
        'n_linear_layers' : 'discrete',
        'n_linear_hidden' : 'discrete',
        'weight_decay' : 'uniform',
        'lambda_param' : 'uniform',
        'epochs' : 'fixed'
}


def sample_uniform(low, high, size): # selects a random number from a uniform distribution
    return np.random.uniform(low, high, size) 


def sample_discrete(low, high, size): # selects a random number from a discrete distribution
    return np.random.randint(low, high+1, size)


def sample_fixed(value, size): # selects a random number from a fixed distribution
    return [value[i] for i in np.random.randint(0, len(value), size)]


def generate_search_space(configs, n_runs=200, param_dict=PARAMETER_DICT, round_decimals=None):

    hyperparameter_list = []
    hyperparameters = np.zeros((n_runs, len(param_dict)), dtype=object)
    cnt = 0
    for key, value in extract_configs(vars(configs), depth=3).items():
        if key in param_dict:
            if isinstance(value, list):
                logger.debug(color_text(f'Creating random samples for parameter `{key}`.', color='blue', style='bold'))
                if len(value) == 1:
                    value = value*2
                if param_dict[key] == 'uniform':
                    hyper = sample_uniform(value[0], value[1], n_runs)
                elif param_dict[key] == 'discrete':
                    hyper = sample_discrete(value[0], value[1], n_runs)
                elif param_dict[key] == 'fixed':
                    hyper = sample_fixed(value, n_runs)
                else:
                    raise ValueError(color_text('Invalid parameter distribution.', color='red', style='bold'))

                if round_decimals is not None:
                    hyperparameters[:, cnt] = np.round(hyper, round_decimals)
                else:
                    hyperparameters[:, cnt] = hyper
            else:
                raise ValueError(f'Parameter {key} is not a list.')
            hyperparameter_list.append(key)
            cnt += 1

    return hyperparameters[:,:len(hyperparameter_list)], hyperparameter_list


