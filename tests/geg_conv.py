from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import (
    add_self_loops,
    get_laplacian,
    remove_self_loops,
)


class GegConv(MessagePassing):
    r"""The Gegenbauer spectral graph convolutional operator that generalizes
    the Chebyshev polynomial convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \sum_{k=1}^{K} \mathbf{Z}^{(k)} \cdot
        \mathbf{\Theta}^{(k)}

    where :math:`\mathbf{Z}^{(k)}` is computed recursively by

    .. math::
        \mathbf{Z}^{(1)} &= \mathbf{X}

        \mathbf{Z}^{(2)} &= \mathbf{\hat{L}} \cdot \mathbf{X}

        k\mathbf{Z}^{(k)} &= 2(k+\alpha-1) \cdot \mathbf{\hat{L}} \cdot
        \mathbf{Z}^{(k-1)} - (k-2\alpha-2)\mathbf{Z}^{(k-2)}

    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian
    :math:`\frac{2\mathbf{L}}{\lambda_{\max}} - \mathbf{I}`.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        alpha (float, optional): The parameter of the Gegenbauer polynomial
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*,
          batch vector :math:`(|\mathcal{V}|)` *(optional)*,
          maximum :obj:`lambda` value :math:`(|\mathcal{G}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        alpha: float = 0, # alpha is the parameter of the Gegenbauer polynomial
        normalization: Optional[str] = 'sym',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                   weight_initializer='glorot') for _ in range(K)
        ])
        self.alpha = alpha if alpha !=0 else alpha + torch.finfo(torch.float).eps

        if bias:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def __norm__(
        self,
        edge_index: Tensor,
        num_nodes: Optional[int],
        edge_weight: OptTensor,
        normalization: Optional[str],
        lambda_max: OptTensor = None,
        dtype: Optional[int] = None,
        batch: OptTensor = None,
    ):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)
        assert edge_weight is not None

        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype,
                                      device=edge_index.device)
        assert lambda_max is not None

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=-1.,
                                                 num_nodes=num_nodes)
        assert edge_weight is not None

        return edge_index, edge_weight

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        lambda_max: OptTensor = None,
    ) -> Tensor:
        """"""
        edge_index, norm = self.__norm__(
            edge_index,
            x.size(self.node_dim),
            edge_weight,
            self.normalization,
            lambda_max,
            dtype=x.dtype,
            batch=batch,
        )

        Gx_0 = x
        out = self.lins[0](Gx_0)

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            Gx_1 = 2*self.alpha*self.propagate(edge_index, x=x, norm=norm, size=None)
            out = out + self.lins[1](Gx_1)*self.norm_term(1)

        for n, lin in enumerate(self.lins[2:], start=2):
            Gx_2 = self.propagate(edge_index, x=Gx_1, norm=norm, size=None)
            Gx_2 = 2*(n+self.alpha-1)/n*Gx_2 - (n+2*self.alpha-2)/n*Gx_0
            out = out + lin.forward(Gx_2)*self.norm_term(n)
            Gx_0, Gx_1 = Gx_1, Gx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def norm_term(self, n):
        return 1/2*(n+self.alpha)/self.alpha \
               if self.alpha == torch.finfo(torch.float).eps else 1

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}, '
                f'alpha={self.alpha}, '
                f'normalization={self.normalization})')


if __name__ == '__main__':
    import torch
    from cheb2_conv import ChebConv2
    from torch_geometric.nn import ChebConv
    from torch_geometric.data import Data
    from torch_geometric.datasets import MNISTSuperpixels

    from scipy import special

    dataset = MNISTSuperpixels(root='/tmp/MNIST')
    data = dataset[0]
    # print(data)
    # edge_index = torch.tensor([[0, 1, 1, 2],
    #                        [1, 0, 2, 1]], dtype=torch.long)
    # x = torch.tensor([[-1], [3], [1]], dtype=torch.float)

    # data = Data(x=x, edge_index=edge_index)

    torch.manual_seed(12345)
    conv1 = GegConv(1, 1, K=10, alpha=0)
    out1 = conv1(data.x, data.edge_index)
    # print(out1.shape)
    print(out1[:5])

    torch.manual_seed(12345)
    conv2 = ChebConv(1, 1, K=10)
    out2 = conv2(data.x, data.edge_index)
    # print(out2.shape)
    print(out2[:5])

    print(out1[:5] - out2[:5])

    # torch.manual_seed(12345)
    # conv1 = GegConv(1, 1, K=3, alpha=1)
    # out1 = conv1(data.x, data.edge_index)
    # # print(out1.shape)
    # print(out1[:5])

    # torch.manual_seed(12345)
    # conv2 = ChebConv2(1, 1, K=3)
    # out2 = conv2(data.x, data.edge_index)
    # # print(out2.shape)
    # print(out2[:5])
