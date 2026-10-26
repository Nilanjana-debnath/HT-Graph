import torch
import torch.nn as nn
import enum
import logging

class LayerType(enum.Enum):
    IMP1 = 0,
    IMP2 = 1,
    IMP3 = 2
class GAT(torch.nn.Module):
 
    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6, layer_type=LayerType.IMP3, log_attention_weights=False):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        GATLayer = get_layer_type(layer_type)  
        num_heads_per_layer = [1] + num_heads_per_layer  


        
        gat_layers = []  
        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  
                num_out_features=num_features_per_layer[i+1],
                num_of_heads=num_heads_per_layer[i+1],
                concat=True if i < num_of_layers - 1 else False,  
                activation=nn.ELU() if i < num_of_layers - 1 else None,  
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )
    
    def forward(self, data):
        rst = self.gat_net(data)
        # print(rst , rst[1].shape, rst[0].shape )
        # if rst[-1] == None:
            # exit()
        return rst


class GATLayer(torch.nn.Module):
 
    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, layer_type, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()
        
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  
        self.add_skip_connection = add_skip_connection
        
        if layer_type == LayerType.IMP1:
            
            self.proj_param = nn.Parameter(torch.Tensor(num_of_heads, num_in_features, num_out_features))
        else:
            
            self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if layer_type == LayerType.IMP1:  
            self.scoring_fn_target = nn.Parameter(self.scoring_fn_target.reshape(num_of_heads, num_out_features, 1))
            self.scoring_fn_source = nn.Parameter(self.scoring_fn_source.reshape(num_of_heads, num_out_features, 1))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        
        self.leakyReLU = nn.LeakyReLU(0.2)  
        self.softmax = nn.Softmax(dim=-1)  
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)
        self.log_attention_weights = log_attention_weights  
        self.attention_weights = None  
        self.init_params(layer_type)

    def init_params(self, layer_type):
        
        nn.init.xavier_uniform_(self.proj_param if layer_type == LayerType.IMP1 else self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  
            self.attention_weights = attention_coefficients

        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:             
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class GATLayerImp4(GATLayer):
    src_nodes_dim = 0  
    trg_nodes_dim = 1  

    nodes_dim = 0      
    head_dim = 1       

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):     
        super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP3, concat, activation, dropout_prob,
                      add_skip_connection, bias, log_attention_weights)
    def forward(self, data):
        in_nodes_features, edge_index, num_of_nodes = data  
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'
        in_nodes_features = self.dropout(in_nodes_features)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)  
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = scores_source, scores_target, nodes_features_proj # currently no list required
        # scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        attentions_per_edge = self.dropout(attentions_per_edge)
        # nodes_features_proj_lifted_weighted = nodes_features_proj_lifted
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)
        out_nodes_features = self.skip_concat_bias(None, in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index, None)

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)
        size = list(exp_scores_per_edge.shape)  
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  
        size[self.nodes_dim] = num_of_nodes  
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)
        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)
        return this.expand_as(other)
    

class GATLayerImp3(GATLayer):
    src_nodes_dim = 0  
    trg_nodes_dim = 1  

    nodes_dim = 0      
    head_dim = 1       

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):     
        super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP3, concat, activation, dropout_prob,
                      add_skip_connection, bias, log_attention_weights)
    def forward(self, data):
        in_nodes_features, edge_index, num_of_nodes = data  
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'
        in_nodes_features = self.dropout(in_nodes_features)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)  
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = scores_source, scores_target, nodes_features_proj # currently no list required
        # scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        attentions_per_edge = self.dropout(attentions_per_edge)
        # nodes_features_proj_lifted_weighted = nodes_features_proj_lifted
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)
        out_nodes_features = self.skip_concat_bias(None, in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index, None)

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)
        size = list(exp_scores_per_edge.shape)  
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  
        size[self.nodes_dim] = num_of_nodes  
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)
        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)
        return this.expand_as(other)


class GATLayerImp2(GATLayer):

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP2, concat, activation, dropout_prob,
                         add_skip_connection, bias, log_attention_weights)

    def forward(self, data):
        in_nodes_features, connectivity_mask = data  
        num_of_nodes = in_nodes_features.shape[0]
        assert connectivity_mask.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={connectivity_mask.shape}.'
        in_nodes_features = self.dropout(in_nodes_features)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)  
        scores_source = torch.sum((nodes_features_proj * self.scoring_fn_source), dim=-1, keepdim=True)
        scores_target = torch.sum((nodes_features_proj * self.scoring_fn_target), dim=-1, keepdim=True)
        scores_source = scores_source.transpose(0, 1)
        scores_target = scores_target.permute(1, 2, 0)
        all_scores = self.leakyReLU(scores_source + scores_target)
        all_attention_coefficients = self.softmax(all_scores + connectivity_mask)
        out_nodes_features = torch.bmm(all_attention_coefficients, nodes_features_proj.transpose(0, 1))
        out_nodes_features = out_nodes_features.permute(1, 0, 2)
        out_nodes_features = self.skip_concat_bias(all_attention_coefficients, in_nodes_features, out_nodes_features)
        return (out_nodes_features, connectivity_mask)

class GATLayerImp1(GATLayer):
    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__(num_in_features, num_out_features, num_of_heads, LayerType.IMP1, concat, activation, dropout_prob,
                         add_skip_connection, bias, log_attention_weights)

    def forward(self, data):
        print(data)
        in_nodes_features, connectivity_mask, num_nodes = data  
        num_of_nodes = in_nodes_features.shape[0]
        assert connectivity_mask.shape == (num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({num_of_nodes},{num_of_nodes}), got shape={connectivity_mask.shape}.'
        in_nodes_features = self.dropout(in_nodes_features)
        nodes_features_proj = torch.matmul(in_nodes_features.unsqueeze(0), self.proj_param)
        nodes_features_proj = self.dropout(nodes_features_proj)  
        scores_source = torch.bmm(nodes_features_proj, self.scoring_fn_source)
        scores_target = torch.bmm(nodes_features_proj, self.scoring_fn_target)
        all_scores = self.leakyReLU(scores_source + scores_target.transpose(1, 2))
        all_attention_coefficients = self.softmax(all_scores + connectivity_mask)
        out_nodes_features = torch.bmm(all_attention_coefficients, nodes_features_proj)
        out_nodes_features = out_nodes_features.transpose(0, 1)
        out_nodes_features = self.skip_concat_bias(all_attention_coefficients, in_nodes_features, out_nodes_features)
        return (out_nodes_features, connectivity_mask)

def resize(e, dim):
    # e:[B, N, D]
    if e.size(-1) >= dim:
        res = e[:, :, :dim]
    else:
        e_padd = torch.zeros(e.size(0), e.size(1), dim-e.size(2)).to(e.device)
        res = torch.cat([e, e_padd], dim=-1)
    return res


class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4, non_linear=True):
    super().__init__()
    #self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

    # special linear layer for motif explainability
    self.non_linear = non_linear
    if not non_linear:
      assert(dim1 == dim2)
      self.fc = nn.Linear(dim1, 1)
      torch.nn.init.xavier_normal_(self.fc1.weight)

  def forward(self, x1, x2):
    z_walk = None
    if self.non_linear:
      x = torch.cat([x1, x2], dim=-1)
      #x = self.layer_norm(x)
      h = self.act(self.fc1(x))
      z = self.fc2(h)
    else: # for explainability
      # x1, x2 shape: [B, M, F]
      x = torch.cat([x1, x2], dim=-2)  # x shape: [B, 2M, F]
      z_walk = self.fc(x).squeeze(-1)  # z_walk shape: [B, 2M]
      z = z_walk.sum(dim=-1, keepdim=True)  # z shape [B, 1]
    return z, z_walk

class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, q, k, v, r_pri=None, mask=None):
        # q, k:  (n*b) x N x dk
        # v:     (n*b) x N x dv
        # r_pri: b x N

        attn = torch.sum(q * k, dim=-1)  # [n*b, N]
        if r_pri is None:
            attn = attn / self.temperature
        else:
            attn = attn.reshape(-1, *r_pri.shape) * r_pri / self.temperature
            attn = attn.reshape(-1, r_pri.size(1))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n*b, N]
        attn = self.dropout(attn)

        output = (attn.unsqueeze(-1) * v).sum(dim=1)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, num_n_type=1, num_e_type=1):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.num_n_type = num_n_type
        self.num_e_type = num_e_type
        
        self.d_r = self.d_model
        self.w_qs = nn.ModuleList()
        self.w_ks = nn.ModuleList()
        self.w_vs = nn.ModuleList()
        for _ in range(num_e_type+1):
            self.w_qs.append(nn.Linear(self.d_r, n_head * d_k, bias=False))
            self.w_ks.append(nn.Linear(self.d_r, n_head * d_k, bias=False))
            self.w_vs.append(nn.Linear(self.d_r, n_head * d_v, bias=False))

        self.relation_pri = nn.Parameter(torch.ones(num_n_type + 1, num_e_type + 1))

        self.attention = ScaledDotProductAttention(d_k ** 0.5, attn_dropout=dropout)
        
        self.fc_lins = nn.ModuleList()
        for _ in range(num_n_type+1):
            self.fc_lins.append(nn.Linear(self.n_head*self.d_v, self.d_model))
        
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameter()

    def reset_parameter(self):
        for i in range(len(self.w_qs)):
            nn.init.xavier_uniform_(self.w_qs[i].weight.data)
            nn.init.xavier_uniform_(self.w_ks[i].weight.data)
            nn.init.xavier_uniform_(self.w_vs[i].weight.data)
        
        for fc in self.fc_lins:
            nn.init.xavier_uniform_(fc.weight.data)

    def _get_relation_pri_of_q(self, seq_utype, seq_etype):
        N = seq_etype.size(1)
        seq = seq_utype.repeat(N, 1).t()
        idx = seq * (self.num_e_type+1) + seq_etype
        pri = self.relation_pri.flatten()
        pri = pri.index_select(0, idx.flatten())
        return pri.view(-1, N)

    def _compute_QKV_by_etype(self, q, k, v, seq_etype):
        for i, etype in enumerate(seq_etype.unique()):
            msk = (seq_etype==etype).unsqueeze(-1)
            if i==0:
                Q = self.w_qs[etype](q * msk)
                K = self.w_ks[etype](k * msk)
                V = self.w_vs[etype](v * msk)
            else:
                Q += self.w_qs[etype](q * msk)
                K += self.w_ks[etype](k * msk)
                V += self.w_vs[etype](v * msk)
        return Q, K, V

    def forward(self, q, k, v, seq_etype=None, seq_utype=None, seq_vtype=None, mask=None):
        """
        params:
            q:    [B, N, D]  of src
            k, v: [B, N, D]  of ngh
            seq_etype: [B, N]
            seq_utype: [B]
            seq_vtype: [B, N]
            mask: [B, N]
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_e = q.size(0), k.size(1)

        # compute Q, K, V in relature feature space
        q, k, v = self._compute_QKV_by_etype(q, k, v, seq_etype)  # (b, N, n*D)

        q = q.view(sz_b, len_e, n_head, d_k)  # (b, N, n_head, D)  源节点到各个边的转移向量
        k = k.view(sz_b, len_e, n_head, d_k)
        v = v.view(sz_b, len_e, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_e, d_k) # (n*b) x N x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_e, d_k) # (n*b) x N x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_e, d_v) # (n*b) x N x dv

        # pri = None
        pri = self._get_relation_pri_of_q(seq_utype, seq_etype)  # [B, N]
        mask = mask.repeat(n_head, 1) # (n*b, N) x .. x ..
        output, attn = self.attention(q, k, v, pri, mask=mask)  # output (n*b) x dv

        output = output.view(n_head, sz_b, d_v)  # n x b x dv
        output = output.transpose(0, 1).contiguous().view(sz_b, -1) # b x (n*dv) = b x d_model

        """ map out to q's feature space """
        out = torch.zeros_like(output)
        for utype in seq_utype.unique():
            mask = seq_utype==utype
            out[mask] = self.fc_lins[utype](output[mask])
        output = out
        
        # [b, d_model]
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output, attn

class AttnModel(nn.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim, edge_dim, time_dim, transfer, n_head=2, dropout=0.1,
                 num_n_type=1, num_e_type=1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          dropout: probability of dropping a neural
          num_n_type: number of node types
          num_e_type: number of edge types
        """
        super(AttnModel, self).__init__()
        
        self.feat_dim = feat_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        
        self.model_dim = (feat_dim + time_dim)
        assert(self.model_dim % n_head == 0)

        self.transfer = transfer
        
        self.norm1, self.norm2 = nn.LayerNorm(feat_dim), nn.LayerNorm(feat_dim)
        
        self.multi_head_target = MultiHeadAttention(n_head, 
                                            d_model=self.model_dim, 
                                            d_k=self.model_dim // n_head, 
                                            d_v=self.model_dim // n_head, 
                                            dropout=dropout,
                                            num_n_type=num_n_type,
                                            num_e_type=num_e_type)
        
        
        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)
        
        
    def forward(self, src, src_t, seq, seq_t, seq_e, seq_etype, seq_utype, seq_vtype, mask):
        """"Attention based temporal attention forward pass
        args:
          src:   float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, 1, Dt], Dt == D
          seq:   float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          seq_etype: long Tensorof shape [B, N], value in (0, 1, ...), 0 is default
          seq_utype: long Tensorof shape [B, N], value in (0, 1, ...), 0 is default
          seq_vtype: long Tensorof shape [B, N], value in (0, 1, ...), 0 is default
          mask:  boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.
        returns:
          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """
        # type mapping
        q, k = self.transfer(src, seq, seq_utype, seq_vtype, seq_etype) # [B, N, D]
        # q, k = src.unsqueeze(1).repeat(1, seq.size(1), 1), seq  # no transfer

        # add & norm
        q = self.norm1(q)  # source node is not associated with edge feature
        k = self.norm2(k + resize(seq_e, self.feat_dim))

        # [B, D+Dt]
        q = torch.cat([q, src_t.repeat(1, q.size(1), 1)], dim=2)
        k = torch.cat([k, seq_t], dim=2)

        # target-attention
        # output: [B, D], attn: [B, N, N]
        output, attn = self.multi_head_target(q=q, k=k, v=k, seq_etype=seq_etype, seq_utype=seq_utype,
                                                 seq_vtype=seq_vtype, mask=mask)
        
        # residual
        # why not use q for residual: src_e is zeros vector and src_t is one vector, so they will not contribute useful information for the source node v0
        output = self.merger(output, src)
        
        return output, attn


def get_layer_type(layer_type):

    assert isinstance(layer_type, LayerType), f'Expected {LayerType} got {type(layer_type)}.'

    if layer_type == LayerType.IMP1:
        return GATLayerImp1
    elif layer_type == LayerType.IMP2:
        return GATLayerImp2
    elif layer_type == LayerType.IMP3:
        return GATLayerImp3
    else:
        raise Exception(f'Layer type {layer_type} not yet supported.')
