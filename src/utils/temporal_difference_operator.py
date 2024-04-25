"""

`temporal_difference_operator.py`
---------------------------------

    Function to compute the temporal difference operator on a signal.

"""

import numpy as np


def temporal_difference_operator(signal_size:int, order:int=1):
    """
    Compute the temporal difference operator on a signal.

    Parameters
    ----------
    signal_size : int
        Size of the signal.
    """

    if order == 1:
        values = [-1, 1]
        n = 0
    elif order == 2:
        values = [-1, 2, -1]
        n = 0
    elif order == 3:
        values = [-1, -1, 4, -1, -1]
        n = 1

    dh = np.zeros((signal_size, signal_size-order-n))
    for i in range(0,signal_size-order-n):
        for j, val in enumerate(values):
            dh[i+j,i] = val

    return dh
