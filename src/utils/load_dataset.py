"""

`load_dataset.py`
----------------

    This script is used to load the data from the processed data.

"""

import os

import h5py
import torch
import numpy as np
from torch_geometric.data import Data

from .tools import generate_sampling_pattern
from .printing import color_text
from .logger import set_logger

logger = set_logger('load_dataset')

class Dataset():
    def __init__(
            self,
            root:str,
            dataset:str,
            sampling_density:float=None,
            transform=None
        ):
        self.root = root
        self.dataset = dataset
        self.sampling_density = sampling_density
        self.transform = transform

    @property
    def scaler(self):
        return self.transform

    def __call__(self):
        with h5py.File(os.path.join(self.root, self.dataset, 'dataset.hdf'), 'r') as f:
            data = np.array(f['data'])
            Dh = torch.tensor(f['Dh'][()], dtype=torch.float)
            L = torch.tensor(f['L'][()], dtype=torch.float)
            W = torch.tensor(f['W'][()], dtype=torch.float)
        raw_data = data.copy()
        edge_index = W.nonzero().T
        edge_attr = W[edge_index[0, :], edge_index[1, :]]

        # if sampling exists, then use it
        if self.sampling_density is not None:
            sampling_pattern = generate_sampling_pattern(
                                        data=data,
                                        pattern='random',
                                        density=self.sampling_density
                                    )
            x = data*sampling_pattern
        else:
            x = data.copy()

        # if transform exists, then use it
        if self.transform is not None:
            self.transform.fit(x.T)
            x = self.transform.transform(x[np.any(x, -1)].T).T
            data = self.transform.transform(data.T).T

        logger.info(color_text('Dataset loaded succesfully', color='green', style='bold'))

        return Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            Dh=Dh,
            raw_data=torch.tensor(raw_data, dtype=torch.float),
            L=L,
            W=W,
            dataset=torch.tensor(data, dtype=torch.float),
            sampling_pattern=torch.tensor(sampling_pattern, dtype=torch.long),
            train_idx=torch.tensor(np.array(np.where(sampling_pattern==1)).T, dtype=torch.long),
            test_idx=torch.tensor(np.array(np.where((sampling_pattern==0) & (raw_data !=0))).T, dtype=torch.long)
        )

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def __str__(self):
        return self.__repr__()