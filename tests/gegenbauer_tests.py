"""
test_geg_conv.py
----------------

    This file contains the unit tests for the GegConv class.

"""
import unittest

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import ChebConv
from torch_geometric.datasets import MNISTSuperpixels

from chebyu_conv import ChebConvU
from src.net.layers.gegen_conv import GegConv
from src.utils.printing import color_text
from src.utils.logger import set_logger

logger = set_logger('tests')

class TestGegConv(unittest.TestCase):

    filter_order = 10
    random_seed = 42

    # Small graph
    edge_index = torch.tensor([[0, 1, 1, 2],
                        [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [3], [1]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    # MNIST dataset
    dataset = MNISTSuperpixels(root='/tmp/MNIST')
    np.random.seed(random_seed)
    idx = np.random.choice(len(dataset))
    mnist_data = dataset[idx]


    def test_small_graph_chebyshev_first_kind(self):
        for k in range(1, self.filter_order+1):
            # we need this to make sure the linear layers are initialized the same way
            torch.manual_seed(self.random_seed)
            conv_geg = GegConv(1, 1, K=k, alpha=0)
            out_geg = conv_geg(self.data.x, self.data.edge_index)

            # we need this to make sure the linear layers are initialized the same way
            torch.manual_seed(self.random_seed)
            conv_cheb = ChebConv(1, 1, K=k)
            out_cheb = conv_cheb(self.data.x, self.data.edge_index)

            self.assertEqual(out_geg.shape, out_cheb.shape, msg='Output shapes do not match')
            self.assertTrue(torch.allclose(out_geg, out_cheb, atol=1e-05), msg='Output values do not match')
            logger.info(color_text(f'MNIST | chebyshev first kind k = {k}. OK', color='green', style='bold'))


    def test_mnist_chebyshev_first_kind(self):
        for k in range(1, self.filter_order+1):
            # we need this to make sure the linear layers are initialized the same way
            torch.manual_seed(self.random_seed)
            conv_geg = GegConv(1, 1, K=k, alpha=0)
            out_geg = conv_geg(self.mnist_data.x, self.mnist_data.edge_index)

            # we need this to make sure the linear layers are initialized the same way
            torch.manual_seed(self.random_seed)
            conv_cheb = ChebConv(1, 1, K=k)
            out_cheb = conv_cheb(self.mnist_data.x, self.mnist_data.edge_index)

            self.assertEqual(out_geg.shape, out_cheb.shape, msg='Output shapes do not match')
            self.assertTrue(torch.allclose(out_geg, out_cheb, atol=1e-05), msg='Output values do not match')
            logger.info(color_text(f'MNIST | chebyshev first kind k = {k}. OK', color='green', style='bold'))


    def test_small_graph_chebyshev_second_kind(self):
        for k in range(1, self.filter_order+1):
            # we need this to make sure the linear layers are initialized the same way
            torch.manual_seed(self.random_seed)
            conv_geg = GegConv(1, 1, K=k, alpha=1)
            out_geg = conv_geg(self.data.x, self.data.edge_index)

            # we need this to make sure the linear layers are initialized the same way
            torch.manual_seed(self.random_seed)
            conv_cheb = ChebConvU(1, 1, K=k)
            out_cheb = conv_cheb(self.data.x, self.data.edge_index)

            self.assertEqual(out_geg.shape, out_cheb.shape, msg='Output shapes do not match')
            self.assertTrue(torch.allclose(out_geg, out_cheb, atol=1e-05), msg='Output values do not match')
            logger.info(color_text(f'small graph | chebyshev second kind k = {k}. OK', color='green', style='bold'))


    def test_mnist_chebyshev_second_kind(self):
        for k in range(1, self.filter_order+1):
            # we need this to make sure the linear layers are initialized the same way
            torch.manual_seed(self.random_seed)
            conv_geg = GegConv(1, 1, K=k, alpha=1)
            out_geg = conv_geg(self.mnist_data.x, self.mnist_data.edge_index)

            # we need this to make sure the linear layers are initialized the same way
            torch.manual_seed(self.random_seed)
            conv_cheb = ChebConvU(1, 1, K=k)
            out_cheb = conv_cheb(self.mnist_data.x, self.mnist_data.edge_index)

            self.assertEqual(out_geg.shape, out_cheb.shape, msg='Output shapes do not match')
            self.assertTrue(torch.allclose(out_geg, out_cheb, atol=1e-05), msg='Output values do not match')
            logger.info(color_text(f'MNIST | chebyshev second kind k = {k}. OK', color='green', style='bold'))


if __name__ == '__main__':
    unittest.main()