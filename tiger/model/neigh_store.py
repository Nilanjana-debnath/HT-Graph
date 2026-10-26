from typing import Dict, List, Tuple, Optional, Union
import torch
from torch import nn, Tensor

class Neigh_store(nn.Module):
    def __init__(self, total_nodes, self_dim, num_neighbors):
        super().__init__()
        # self.n = n
        # self.dim = dim
        self.num_neighbors = num_neighbors
        self.total_nodes = total_nodes
        self.num_raw = 3
        self.self_dim = self_dim
        # self.register_buffer('neighborhood_store', torch.zeros(n, dim), persistent=True)
        # self.register_buffer('self_rep', torch.zeros(n), persistent=True)
        # self.register_buffer('prev_raw', torch.zeros(n).bool(), persistent=True)
    
        
    def clear_store(self):
        self.neighborhood_store[...] = None

    def reset_store(self):
        ngh_stores = []
        for i in self.num_neighbors:
            max_e_idx = self.total_nodes * i
            raw_store = torch.zeros(max_e_idx, self.num_raw)
            hidden_store = torch.empty(max_e_idx, self.ngh_dim)
            ngh_store = torch.cat((raw_store, nn.init.xavier_uniform_(hidden_store)), -1).to(self.device)
            ngh_stores.append(ngh_store)
        # self.neighborhood_store = ngh_stores
        # self.self_rep = torch.zeros(self.total_nodes, self.self_dim).to(self.device)
        # self.prev_raw = torch.zeros(self.total_nodes, 3).to(self.device)
        self.register_buffer('neighborhood_store', ngh_stores, persistent=True)
        self.register_buffer('self_rep', torch.zeros(self.total_nodes, self.self_dim), persistent=True)
        self.register_buffer('prev_raw', torch.zeros(self.total_nodes, 3), persistent=True)

    def get_neighborhood_store(self):
        return self.neighborhood_store[...]

    def set_neighborhood_store(self, neighborhood_store):
        self.neighborhood_store.data[...] = neighborhood_store


    