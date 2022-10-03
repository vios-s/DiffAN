from typing import List, Optional, Type
import torch
from torch import nn
from torch.nn import Dropout, LayerNorm, Linear, Module, Sequential, BatchNorm1d

class DiffMLP(Module):
    def __init__(self, n_nodes: int) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        big_layer = max(1024, 5*self.n_nodes)
        small_layer = max(128, 3*self.n_nodes)
        
        self.main_block = nn.Sequential(
            nn.Linear(self.n_nodes+1,small_layer, bias= False),
            nn.LeakyReLU(),
            nn.LayerNorm([small_layer]),
            nn.Dropout(0.2),
            nn.Linear(small_layer, big_layer),
            nn.LeakyReLU(),
            nn.LayerNorm([big_layer]),
            nn.Linear(big_layer,big_layer),
            nn.LeakyReLU(),
            nn.Linear(big_layer,small_layer),
            nn.LeakyReLU(),
            nn.Linear(small_layer,self.n_nodes),
        )

    def forward(self, X, t):
        X_t = torch.cat([X,t.unsqueeze(1)],axis = 1)
        return self.main_block(X_t)
