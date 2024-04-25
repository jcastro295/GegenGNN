import torch
from torch_scatter import scatter_mean


class NeighborNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.Tensor(num_features),
                                        requires_grad=True)
        self.beta = torch.nn.Parameter(torch.Tensor(num_features),
                                       requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.gamma)
        torch.nn.init.zeros_(self.beta)

    def forward(self, x, edge_index):
        row, col = edge_index
        mean = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))
        mean = torch.mean(mean, dim=-1, keepdim=True)
        var = scatter_mean((x[col] - mean[row])**2,
                           row,
                           dim=0,
                           dim_size=x.size(0))
        var = torch.mean(var, dim=-1, keepdim=True)
        # std = scatter_std(x[col], row, dim=0, dim_size=x.size(0))
        out = (x[col] - mean[row]) / (var[row] + self.eps).sqrt()
        # out = (x[col] - mean[row]) / (std[row]**2 + self.eps).sqrt()
        out = self.gamma * out + self.beta

        return out
