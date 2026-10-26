from typing import Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FeatureGetter(nn.Module):
    """
    The base class of feature getters.
    """
    n_nodes: int
    n_edges: int
    nfeat_dim: int
    efeat_dim: int
    ngh_feat_dim: int
    out_dim: int
    num_e_type: int
    device: torch.device

    def get_node_embeddings(self, nids: Tensor) -> Tensor:
        raise NotImplementedError

    def get_edge_embeddings(self, eids: Tensor) -> Tensor:
        raise NotImplementedError
    

class NumericalFeature(nn.Module):
    def __init__(self, nfeats: Optional[Tensor], efeats: Optional[Tensor],
                 dim: int, ngh_feat_dim:int , time_dim:int, pos_dim:int, self_dim:int, num_n_type:int, num_e_type:int, *, use_tsfm: bool=False,
                 register_buffer: bool=True, device: torch.device=None):
        """
        Get node/edge (transformed) features.
        ---
        nfeats: node features (could be None)
        efeats: edge features (could be None)
        dim: the output dim
        ngh_feat_dim: the output dim
        num_e_type: edge_types
        use_tsfm: if true, do liner projection on the raw features
        register_buffer: if true, features are registered as buffers
        device: torch device
        """
        super().__init__()
        self.pin_mem = register_buffer
        self.device = device
        self.use_tsfm = use_tsfm
        
        # Node feats
        if nfeats is None:
            self.n_nodes = None
            self.nfeat_dim = None
            self.nfeats = None
        else:
            self.n_nodes, self.nfeat_dim = nfeats.shape
            if self.pin_mem:  # can be auto moved to cuda
                self.register_buffer('nfeats', nfeats, persistent=False)
            else:
                self.nfeats = nfeats
        
        # Edge feats
        if efeats is None:
            self.n_edges = None
            self.efeat_dim = None
            self.efeats = efeats
        else:
            self.n_edges, self.efeat_dim = efeats.shape
            if self.pin_mem:
                self.register_buffer('efeats', efeats, persistent=False)
            else:
                self.efeats = efeats
        
        self.out_dim = dim

        if self.use_tsfm:
            if self.nfeats is not None:
                self.node_linear = nn.Linear(self.nfeat_dim, self.out_dim)
            if self.efeats is not None:
                self.edge_linear = nn.Linear(self.efeat_dim, self.out_dim)
        
        self.nfeat_dim = self.nfeat_dim if self.nfeat_dim else self.out_dim
        self.efeat_dim = self.efeat_dim if self.efeat_dim else self.out_dim
        self.ngh_feat_dim = ngh_feat_dim
        self.num_e_type = num_e_type
        self.num_n_type = num_n_type
        self.pos_dim = pos_dim
        self.time_dim = time_dim
        self.self_dim = self_dim
        
    
    def get_node_embeddings(self, nids: Tensor) -> Tensor:
        if self.nfeats is None:
            shape = nids.shape
            zeros = torch.zeros(shape, device=nids.device)
            zeros = zeros.unsqueeze(-1).expand(*shape, self.out_dim)
            return zeros
        if not self.pin_mem:
            x = F.embedding(nids.to(self.nfeats.device), self.nfeats).to(self.device)
        else:
            x = F.embedding(nids, self.nfeats)
        if self.use_tsfm:
            x = self.node_linear(x)
        return x

    def get_edge_embeddings(self, eids: Tensor) -> Tensor:
        if self.efeats is None:
            shape = eids.shape
            zeros = torch.zeros(shape, device=eids.device)
            zeros = zeros.unsqueeze(-1).expand(*shape, self.out_dim)
            return zeros
        if not self.pin_mem:
            x = F.embedding(eids.to(self.efeats.device), self.efeats).to(self.device)
        else:
            x = F.embedding(eids, self.efeats)
        if self.use_tsfm:
            x = self.edge_linear(x)
        return x
