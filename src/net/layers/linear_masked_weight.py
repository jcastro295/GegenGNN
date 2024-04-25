from torch.functional import F
from torch.nn import Linear


class LinearMaskedWeight(Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearMaskedWeight, self).__init__(in_features, out_features, bias)
        self.weight.requires_grad =False
        self.bias.requires_grad = False

    def forward(self, input, mask):
        maskedW = self.weight*mask
        return F.linear(input, maskedW, self.bias)