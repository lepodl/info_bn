from torch import nn
import torch.nn.functional as F
from mup import MuReadout

class MLP(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=3072, out_dim=1, nonlin=F.relu, output_mult=1.0, input_mult=1.0, param="ntk"):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.param = param
        self.fc_1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        if self.param == "muP":
            self.fc_3 = MuReadout(hidden_dim, out_dim, bias=False, output_mult=self.output_mult)
            self.reset_parameters()
        elif self.param == "ntk":
            self.fc_3 = nn.Linear(hidden_dim, out_dim, bias=False)
            self.reset_parameters()
        elif self.param == "sp":
            self.reset_parameters()
        else:
            raise NotImplementedError

    def reset_parameters(self):
        if self.param == "muP":
            nn.init.kaiming_normal_(self.fc_1.weight, a=1, mode='fan_in')
            self.fc_1.weight.data /= self.input_mult ** 0.5
            nn.init.kaiming_normal_(self.fc_2.weight, a=1, mode='fan_in')
            nn.init.zeros_(self.fc_3.weight)
        elif self.param == "ntk":
            nn.init.normal_(self.fc_1.weight, std=0.3)
            nn.init.normal_(self.fc_2.weight, std=0.3)
            nn.init.normal_(self.fc_3.weight, std=0.3)

    def forward(self, x):
        if self.param == "muP":
            out = self.nonlin(self.fc_1(x) * self.input_mult ** 0.5)
            out = self.nonlin(self.fc_2(out))
            return self.fc_3(out)
        elif self.param == "ntk":
            out = self.nonlin(self.fc_1(x) / self.input_dim ** 0.5)
            out = self.nonlin(self.fc_2(out) / self.hidden_dim ** 0.5)
            out = self.fc_3(out) / self.hidden_dim ** 0.5
            return out
        else:
            out = self.nonlin(self.fc_1(x))
            out = self.nonlin(self.fc_2(out))
            out = self.fc_3(out)
            return out
