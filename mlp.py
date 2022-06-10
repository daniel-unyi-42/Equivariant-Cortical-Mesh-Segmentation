import torch

from torch.nn import Module, Parameter, ReLU, Linear, BatchNorm1d

class MLP(Module):

    def __init__(self, device, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.lin0 = Linear(in_dim, hidden_dim)
        self.act0 = ReLU()
        self.batchnorm0 = BatchNorm1d(hidden_dim)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.act1 = ReLU()
        self.batchnorm1 = BatchNorm1d(hidden_dim)
        self.lin2 = Linear(hidden_dim, hidden_dim)
        self.act2 = ReLU()
        self.batchnorm2 = BatchNorm1d(hidden_dim)
        self.lin3 = Linear(hidden_dim, hidden_dim)
        self.act3 = ReLU()
        self.batchnorm3 = BatchNorm1d(hidden_dim)
        self.lin4 = Linear(hidden_dim, hidden_dim)
        self.act4 = ReLU()
        self.batchnorm4 = BatchNorm1d(hidden_dim)
        self.lin5 = Linear(hidden_dim, out_dim)
        self.device = device
        self.to(device)

    def forward(self, data):
        data = data.to(self.device)
        x  = torch.cat([data.x, data.pos], axis=1)
        x = self.act0(self.lin0(x))
        x = self.batchnorm0(x)
        x = self.act1(self.lin1(x))
        x = self.batchnorm1(x)
        x = self.act2(self.lin2(x))
        x = self.batchnorm2(x)
        x = self.act3(self.lin3(x))
        x = self.batchnorm3(x)
        x = self.act4(self.lin4(x))
        x = self.batchnorm4(x)
        return self.lin5(x)
