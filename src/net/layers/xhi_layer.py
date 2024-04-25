import torch
from torch.nn.utils import spectral_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils.get_laplacian import get_laplacian

from src.net.layers import LinearMaskedWeight


class XhiLayer(MessagePassing):
    def __init__(
            self,
            in_channels,
            out_channels,
            k=3,
            reservoir_act_fun = lambda x: x,
            device='cpu'
            ):
        super(XhiLayer, self).__init__()
        self.k = k
        self.lin = spectral_norm(LinearMaskedWeight(in_channels*k, out_channels*k)) #SPECTRAL NORM
        self.reservoir_act_fun = reservoir_act_fun
        self.xhi_layer_mask = self.get_xhi_layer_mask(in_channels, out_channels).to(device)
        self.bn_hidden_rec = torch.nn.BatchNorm1d(out_channels*k)


    def get_xhi_layer_mask(self, in_channels, out_channels):

        xhi_layer_mask = []
        for i in range(self.k):
            mask_ones = torch.ones(out_channels, in_channels*(i + 1))
            mask_zeros=torch.zeros(out_channels, in_channels*(self.k-(i+1)))
            xhi_layer_mask.append(torch.cat([mask_ones,mask_zeros],dim=1))

        return torch.cat(xhi_layer_mask, dim=0)


    def forward(self, x, edge_index, edge_weight=None):

        #compute Laplacian
        L_edge_index, L_values = get_laplacian(edge_index, normalization="sym")
        L = torch.sparse.FloatTensor(L_edge_index,
                                     L_values,
                                     torch.Size([x.shape[0],x.shape[0]])).to_dense()

        h = [x]
        for i in range(self.k-1):
            xhi_layer_i = torch.mm(torch.matrix_power(L,i+1),x)
            h.append(xhi_layer_i)

        h = self.lin(torch.cat(h, dim=1), self.xhi_layer_mask)
        h = self.reservoir_act_fun(h)
        h = self.bn_hidden_rec(h)

        #h_avg = global_mean_pool(h.unsqueeze(2), None, x.shape[0])
        #h_add = global_add_pool(h, None, x.shape[0])
        #h_max = global_max_pool(h, None, x.shape[0])

        #h = torch.cat([h_avg, h_add, h_max], dim=1)

        return h