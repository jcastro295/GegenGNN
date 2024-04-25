import torch
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing

class Prop(MessagePassing):
    def __init__(self, in_channels, K, bias=False, **kwargs):
        super(Prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.proj = Linear(in_channels, 1, bias=bias)

    def forward(self, x, edge_index, norm):

        preds = []
        preds.append(x)
        for _ in range(self.K):
            x = self.propagate(
                edge_index,
                x=x,
                norm=norm
                )
            preds.append(x)

        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)

    def reset_parameters(self):
        self.proj.reset_parameters()
