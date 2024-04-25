"""

`laplacian_regularizer.py`
--------------------------

    This file contains the laplacian regularizer

"""

import torch

def laplacian_regularizer(output, Dh, L, lambda_param):
    """
    This function computes the laplacian regularizer
    
    Parameters
    ----------
    output: torch.Tensor
        The output of the model
    Dh: torch.Tensor
        The difference matrix
    L: torch.Tensor
        The laplacian matrix
    lambda_param: float
        The regularization parameter

    """

    difference_signal = torch.mm(output, Dh)
    first_term = torch.mm(torch.transpose(difference_signal, 0, 1), L)

    return lambda_param*torch.trace(torch.mm(first_term, difference_signal))