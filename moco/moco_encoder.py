import torch
import torch.nn as nn

from encoder import Simple3DEncoder as C3D          #  Conv3D
from tcn import TemporalConvNet as TCN              #  TCN

import os


class C3D_TCN(nn.Module):

    def __init__(self, tcn_out_channel=64, c3d_path='', tcn_path=''):
        super(C3D_TCN, self).__init__()

        self.c3d = C3D(in_channels=3) 
        self.tcn = TCN(245760, [128,128,64,tcn_out_channel]) # 245760 == 128, 983040 == 256, 384000 == 160

        self.load_models(c3d_path, tcn_path)

    def load_models(self, c3d_path, tcn_path):
        if os.path.exists(c3d_path):
            self.c3d.load_state_dict(torch.load(c3d_path))
        if os.path.exists(tcn_path):
            self.tcn.load_state_dict(torch.load(tcn_path))

    def save_models(self, c3d_path, tcn_path):
        torch.save(self.c3d.state_dict(), c3d_path)
        torch.save(self.tcn.state_dict(), tcn_path)
    
    def forward(self, X):
        N, WC, RGB, F, W, H = X.shape
        shape = [N*WC, RGB, F, W, H]

        X = self.c3d(X.reshape(shape))

        shape = [N, WC, -1]
        X = X.reshape(shape)

        X = torch.transpose(X, 1, 2)
        X = self.tcn(X)
        X = torch.transpose(X, 1, 2)

        shape = [N, -1]
        X = X.reshape(shape)

        return X