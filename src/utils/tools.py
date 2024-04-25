"""

`tools.py`
----------

    A module to get tools for neural networks

"""

import os
import inspect
import random
from typing import Union, Tuple
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

import src.net.models as models
from src.utils.laplacian_regularizer import laplacian_regularizer
from src.utils.logger import set_logger
from src.utils.printing import color_text
from src.configs.configs_handler import Configs


logger = set_logger('tools')

def train(
        model:nn.Module,
        data:Data,
        optimizer:optim.Optimizer,
        criterion:nn.Module,
        epoch:int,
        configs:Union[Configs, SimpleNamespace]):

    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss_error = criterion(output[data.train_idx[:,0], data.train_idx[:,1]],
                            data.dataset[data.train_idx[:,0], data.train_idx[:,1]])
    loss_regularization = laplacian_regularizer(output, data.Dh, data.Le, configs.lambda_param)
    loss_train = loss_error + loss_regularization
    loss_train.backward()
    optimizer.step()

    if configs.verbose['train']:
        logger.info(color_text(f"{epoch+1}/{configs.epochs} Training : loss train = " +\
            f"{loss_train.item():.4f}, loss error = {loss_error.item():.4f}, " +\
            f"loss reg = {loss_regularization.item():.4f}, " + f"lr = {optimizer.param_groups[0]['lr']:.4f}",
            color='gray', style='bold'))

    return loss_train.item()


def validation(
        model:nn.Module, 
        data:Data,
        criterion:nn.Module,
        configs:Union[Configs, SimpleNamespace]):
    model.eval()
    output = model(data)
    loss_val = criterion(output[data.val_idx[:, 0], data.val_idx[:, 1]],
                        data.dataset[data.val_idx[:, 0], data.val_idx[:, 1]])
    if configs.verbose['val']:
        logger.info(color_text("Validation : loss = {:.4f}".format(loss_val.item()),
                               color='gray', style='bold'))

    return loss_val.item()


def test(
        model:nn.Module,
        data:Data,
        criterion:nn.Module,
        configs:Union[Configs, SimpleNamespace]):
    model.eval()

    output = model(data)
    loss_test = criterion(output[data.test_idx[:, 0], data.test_idx[:, 1]],
                          data.dataset[data.test_idx[:, 0], data.test_idx[:, 1]])
    mae_loss = F.l1_loss(output[data.test_idx[:, 0], data.test_idx[:, 1]],
                         data.dataset[data.test_idx[:, 0], data.test_idx[:, 1]])
    mape_loss = np.mean(np.abs((data.dataset[data.test_idx[:, 0], data.test_idx[:, 1]].detach().cpu().numpy()  \
                        - output[data.test_idx[:, 0], data.test_idx[:, 1]].detach().cpu().numpy() ) \
                        / data.dataset[data.test_idx[:, 0], data.test_idx[:, 1]].detach().cpu().numpy())) * 100
    if configs.verbose['test']:
        logger.info(color_text("Test : loss = {:.4f}".format(loss_test.item()), color='gray', style='bold'))

    return loss_test.item(), mae_loss.item(), mape_loss, output.detach().cpu().numpy().astype(float)


def generate_sampling_pattern(data, pattern='random', density:Union[int,float]=0.1,
                              indices:Union[list,np.ndarray]=None) -> np.ndarray:
    """
    Generate sampling pattern.
    Parameters
    ----------
    data : np.ndarray
        Data matrix.
    pattern : str, optional (default='random')
        Sampling pattern.
    density : Union[int,float], optional (default=0.1)
        Sampling density. 
    indices : Union[list,np.ndarray], optional (default=None)
        Indices to create pattern for certain nodes and times.
    Returns
    -------
    random_pattern : np.ndarray
        Sampling pattern.
    """

    n_nodes, signal_t = data.shape

    # sanity check of data
    valid_idx = data != 0

    while True:
        random_pattern = np.zeros((n_nodes, signal_t))

        if pattern =='random':
            axis = 1
            n_samples = np.round(density*n_nodes).astype(int)
            for i in range(signal_t):
                filtered_idx = np.asarray(valid_idx[:,i]).nonzero()[0]
                idx = np.random.choice(filtered_idx, n_samples, replace=False)
                random_pattern[idx, i] = 1
        elif pattern == 'nodes':
            axis = 0
            random_pattern[indices, :] = 1
        elif pattern == 'time':
            axis = 0
            random_pattern[:, indices] = 1
        else:
            NotImplementedError('The sampling pattern `%s` is not implemented', pattern)

        if np.all(np.sum(random_pattern, axis=axis)):
            break
        else:
            logger.warning(color_text('The sampling pattern is not valid. Trying again.', color='yellow', style='bold'))

    return random_pattern


def parse_model_name(model_name:str)-> str:
    model_name = model_name.lower()

    if model_name in ['gegen', 'gegenbauer', 'gegennet']:
        return 'gegen'
    elif model_name in ['cheby', 'chebyshev', 'chebynet']:
        return 'cheby'
    elif model_name in ['gcn', 'gcnnet']:
        return 'gcn'
    elif model_name in ['gat', 'gatnet']:
        return 'gat'
    elif model_name in ['transformer', 'transformernet']:
        return 'transformer'
    elif model_name in ['hdla', 'gcnhdla', 'gcn-hdla']:
        return 'hdla'
    elif model_name in ['ffk', 'ffkgcnii', 'ffk-gcnii']:
        return 'ffk'
    elif model_name in ['mr', 'mrgnn', 'mr-gnn']:
        return 'mr'
    else:
        raise NotImplementedError('The model `%s` is not implemented', model_name)


def get_data_folder(data_folder:str, model:str='gegen') -> str:
    if model in ['gegen', 'cheby', 'gcn', 'hdla', 'ffk', 'mr']:
        return os.path.join(data_folder, 'polynomial')
    elif model in ['gat', 'transformer']:
        return os.path.join(data_folder, 'attention')
    else:
        raise ValueError(f'Unknown model {model}.')


def get_model(model_name:str, data:Data, configs) -> Tuple[str, torch.nn.Module]:
    model_name = model_name.lower()

    if model_name == 'gegen':
        return (inspect.getfile(models.CascadedGeg),
                models.CascadedGeg(
                    n_features=data.x.shape[1],
                    n_hidden_conv=configs.n_conv_hidden,
                    n_hidden_linear=configs.n_linear_hidden,
                    n_classes=data.dataset.shape[1],
                    n_conv_layers=configs.n_conv_layers,
                    n_linear_layers=configs.n_linear_layers,
                    filter_order=configs.filter_order,
                    alpha=configs.alpha,
                    dropout=configs.dropout,
                    **configs.model_kwargs
                    )
        )

    elif model_name == 'cheby':
        return (inspect.getfile(models.ParameterizedCheby),
                models.ParameterizedCheby(
                    n_features=data.x.shape[1],
                    n_hidden=configs.n_conv_hidden,
                    n_classes=data.dataset.shape[1],
                    n_layers=configs.n_conv_layers,
                    filter_order=configs.filter_order,
                    dropout=configs.dropout,
                    **configs.model_kwargs
                    )
        )

    elif model_name == 'gcn':
        return (inspect.getfile(models.ParameterizedGCN), 
                models.ParameterizedGCN(
                    n_features=data.x.shape[1],
                    n_hidden=configs.n_conv_hidden,
                    n_classes=data.dataset.shape[1], 
                    n_layers=configs.n_conv_layers,
                    dropout=configs.dropout,
                    **configs.model_kwargs
                    )
        )

    elif model_name == 'gat':
        return (inspect.getfile(models.ParameterizedGAT), 
                models.ParameterizedGAT(
                    n_features=data.x.shape[1],
                    n_hidden=configs.n_conv_hidden,
                    n_classes=data.dataset.shape[1], 
                    n_layers=configs.n_conv_layers,
                    n_heads=configs.heads,
                    dropout=configs.dropout,
                    **configs.model_kwargs
                    )
        )

    elif model_name == 'transformer':
        return (inspect.getfile(models.ParameterizedTransformer), 
                models.ParameterizedTransformer(
                    n_features=data.x.shape[1],
                    n_hidden=configs.n_conv_hidden,
                    n_classes=data.dataset.shape[1], 
                    n_layers=configs.n_conv_layers,
                    n_heads=configs.heads,
                    dropout=configs.dropout,
                    **configs.model_kwargs
                    )
        )

    elif model_name == 'hdla':
        return (inspect.getfile(models.GCNDHLA),
                models.GCNDHLA(
                    n_features=data.x.shape[1],
                    n_hidden=configs.n_conv_hidden,
                    n_classes=data.dataset.shape[1], 
                    n_layers=configs.n_conv_layers,
                    dropout=configs.dropout,
                    norm=configs.layer_normalization,
                )
        )

    elif model_name == 'ffk':
        return (inspect.getfile(models.DAGNN),
                models.DAGNN(
                    n_features=data.x.shape[1],
                    n_hidden=configs.n_conv_hidden,
                    n_classes=data.dataset.shape[1], 
                    n_layers=configs.n_conv_layers,
                    dropout=configs.dropout,
                    k=configs.k,
                    **configs.model_kwargs
                )
        )

    elif model_name == 'mr':
        return (inspect.getfile(models.MRGNN),
                models.MRGNN(
                    n_features=data.x.shape[1],
                    n_hidden=configs.n_conv_hidden,
                    n_classes=data.dataset.shape[1], 
                    n_layers=configs.n_conv_layers,
                    dropout=configs.dropout,
                    k=configs.k,
                    output=configs.output,
                    device=configs.device,
                    **configs.model_kwargs
                )
        )

    else:
        raise KeyError("Sorry, but that model isn't available")


def get_optimizer(optimizer_name, params, **kwargs):
    """
    Optimizer for training
    Parameters:
    -----------
    optimizer_name: String
        Name of optimizer. Following options are supported.
    Returns:
    --------
    optimizer: Obj
        Selected optimizer
    """

    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        optimizer = optim.Adam(params, **kwargs)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(params, **kwargs)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(params, **kwargs)
    elif optimizer_name == "adamax":
        optimizer = optim.Adamax(params, **kwargs)
    else:
        raise KeyError("Sorry, but that optimizer isn't available")

    return optimizer


def get_learning_rate_scheduler(scheduler_name, optimizer, **kwargs):
    """
    Leaning rate options for training
    Parameters:
    -----------
    scheduler_name: String
        Name of the learning rate scheduler. Supported:
        "cosine"        - the cosine annealing scheduler
        "exponential"   - an exponential decay scheduler
        "step"          - a step scheduler (needs to be configured
                        through kwargs)
        "plateau"       - the plateau scheduler
        "none"          - disables scheduling by initializing a step
                        scheduler that never actually decreases the
                        learning rate
    optimizer: Obj
        Current optimizer
    Returns:
    --------
    scheduler: Obj
        Selected scheduler
    """

    scheduler_name = scheduler_name.lower()

    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler_name == "none":
        scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=1, **kwargs)
    else:
        raise KeyError("Sorry, but that scheduler isn't available")
    return scheduler


def get_criterion(criterion_name, **kwargs):
    """
    Set the criterium for training
    Parameters:
    -----------
    criterion_name: String
        Name of criterion. Supported:
        "crossentropy"  - cross entropy loss (classification)
        "mse"           - mean squared error loss (regression)
        "l1"            - L1 loss (regression)
    Return:
    -------
    criterion: Obj
        Selected criterion
    """
    criterion_name= criterion_name.lower()
    if criterion_name == "crossentropy":
        criterion = nn.CrossEntropyLoss(**kwargs)
    elif criterion_name == "mse":
        criterion = nn.MSELoss(**kwargs)
    elif criterion_name == 'l1': 
        criterion = nn.L1Loss(**kwargs)
    else:
        raise KeyError(
            "Sorry, but that criterion isn't available. Please add it or check your spelling.")
    return criterion


def get_scaler(scaler:str):
    """
    Set the scaler for training
    Parameters:
    -----------
    scaler: String
        Name of scaler. Supported:
        "standard"  - StandardScaler
        "minmax"    - MinMaxScaler
        "robust"    - RobustScaler
    Return:
    -------
    scaler: Obj
        Selected scaler
    """
    scaler = scaler.lower()
    if scaler == "standard":
        scaler = StandardScaler()
    elif scaler == "minmax":
        scaler = MinMaxScaler()
    elif scaler == "robust":
        scaler = RobustScaler()
    elif scaler == "none":
        scaler = None
    else:
        raise KeyError(
            "Sorry, but that scaler isn't available. Please add it or check your spelling.")
    return scaler


def set_device(cuda:bool=False, gpu:int=0):
    """
    Get the device for training
    Return:
    -------
    device: Obj
    """

    if cuda:
        if torch.cuda.device_count() > gpu:
            device = torch.device('cuda', gpu)
            logger.debug(color_text(f'Setting up GPU {gpu}...', color='gray', style='bold'))
        else:
            logger.debug(color_text(f'GPU {gpu} not available, setting up GPU 0...', color='gray', style='bold'))
            device = torch.device('cuda', 0)
    else:
        logger.debug(color_text('GPU not available, setting up CPU...', color='gray', style='bold'))
        device = torch.device('cpu')

    return device


def set_seed(seed:int=42):
    """
    Set the seed for training

    Parameters:
    -----------
    seed: int
        Seed for training. Default: 42
    
    Return:
    -------
    None
    """

    logger.debug(color_text(f'Setting seed to {seed}...', color='gray', style='bold'))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def ffkgcn_preprocess_adj(adj, k, gamma):
    """
    Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1), dtype=np.float32)
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_add = adj
    k_adj = np.power(gamma, k) * sp.eye(adj.shape[0]) + np.power(gamma, k-1) * adj

    for i in range(k-1):
        adj_add = sp.csr_matrix(adj_add).dot(d_mat_inv_sqrt).dot(adj)
        k_adj += np.power(gamma, k-2-i)*adj_add
    adj_normalized = normalize_adj(k_adj)

    return adj_normalized.A


def ffkgcn_norm_data(data:Data, configs:Configs):
    """
    Runs the renormalization trick from:
    Graph Neural Networks with High-Order Polynomial Spectral Filters
    """

    num_nodes = int(data.edge_index.max()) + 1
    adj = np.zeros([num_nodes, num_nodes])

    for item in data.edge_index.t().tolist():
        a = item[0]
        b = item[1]
        if a == b:
            continue
        else:
            adj[a, b] += 1
    re_adj = ffkgcn_preprocess_adj(adj, configs.k, configs.gamma_norm)
    edg = list()
    weight = list()
    for i in range(re_adj.shape[0]):
        for j in range(re_adj.shape[1]):
            if re_adj[i, j] > 0:
                edg.append((i, j))
                weight.append(re_adj[i, j])
    edge_index = torch.LongTensor(edg).t()
    norm = torch.FloatTensor(weight)
    data.edge_index = edge_index
    data.edge_attr = norm

    return data
