# Public Packages
import torch
import torch.nn as nn
import math

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

# Relation Network Module
class RelationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv1d(input_size*2,input_size,kernel_size=1),
                        nn.BatchNorm1d(input_size, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv1d(input_size,input_size,kernel_size=3),
                        nn.BatchNorm1d(input_size, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(2))
        self.fc1 = nn.Linear(79, hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

        # # Initialize itself
        self.apply(weights_init)

    def forward(self,x):    
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = nn.functional.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

class RelationNetworkZero(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RelationNetworkZero, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv1d(input_size, int(input_size/2), kernel_size=1),
                        nn.BatchNorm1d(int(input_size/2), momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv1d(int(input_size/2), int(input_size/4), kernel_size=3),
                        nn.BatchNorm1d(int(input_size/4), momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool1d(2))
        self.fc1 = nn.Linear(79, hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

        # # Initialize itself
        self.apply(weights_init)

    def forward(self,x):    
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = nn.functional.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out
