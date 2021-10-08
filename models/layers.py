import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax, degree, to_dense_batch
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_geometric.nn.inits import glorot, zeros
import math


### Hypergraph convolution for message passing on Dual Hypergraph 
class HypergraphConv(MessagePassing):

    def __init__(self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HypergraphConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_channels, heads * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def message(self, x_j, edge_index_i, norm, alpha):
        out = norm[edge_index_i].view(-1, 1, 1) * x_j.view(-1, self.heads, self.out_channels)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out
        return out

    def forward(self, x, hyperedge_index, hyperedge_weight=None):
        
        x = torch.matmul(x, self.weight)
        alpha = None

        if self.use_attention:
            x = x.view(-1, self.heads, self.out_channels)
            x_i, x_j = x[hyperedge_index[0]], x[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if hyperedge_weight is None:
            D = degree(hyperedge_index[0], x.size(0), x.dtype)
        else:
            D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                            hyperedge_index[0], dim=0, dim_size=x.size(0))
        D = 1.0 / D
        D[D == float("inf")] = 0

        if hyperedge_index.numel() == 0:
            num_edges = 0
        else:
            num_edges = hyperedge_index[1].max().item() + 1
        B = 1.0 / degree(hyperedge_index[1], num_edges, x.dtype)
        B[B == float("inf")] = 0
        if hyperedge_weight is not None:
            B = B * hyperedge_weight

        num_nodes = x.size(0)
        dif = max([num_nodes, num_edges]) - num_nodes # get size of padding
        x_help = F.pad(x, (0,0,0, dif)) # create dif many nodes

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x_help, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)

        out = out[:num_nodes] # prune back to original x.size()

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


### Multi-head Attention Block used for GMPool
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, cluster=False, conv=None):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k, self.fc_v = self.get_fc_kv(dim_K, dim_V, conv)
        
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.softmax_dim = 2
        if cluster == True:
            self.softmax_dim = 1

    def forward(self, Q, K, attention_mask=None, graph=None, return_attn=False):
        Q = self.fc_q(Q)

        # Adj: Exist (graph is not None), or Identity (else)
        if graph is not None:

            (x, edge_index, batch) = graph

            K, V = self.fc_k(x, edge_index), self.fc_v(x, edge_index)

            K, _ = to_dense_batch(K, batch)
            V, _ = to_dense_batch(V, batch)

        else:

            K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
            A = torch.softmax(attention_mask + attention_score, self.softmax_dim)
        else:
            A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), self.softmax_dim)

        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        if return_attn:
            return O, A
        else:
            return O

    def get_fc_kv(self, dim_K, dim_V, conv):

        if conv == 'GCN':

            fc_k = GCNConv(dim_K, dim_V)
            fc_v = GCNConv(dim_K, dim_V)

        elif conv == 'Hyper':

            fc_k = HypergraphConv(dim_K, dim_V)
            fc_v = HypergraphConv(dim_K, dim_V)

        else:

            fc_k = nn.Linear(dim_K, dim_V)
            fc_v = nn.Linear(dim_K, dim_V)

        return fc_k, fc_v


class GMPool(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False, cluster=False, mab_conv=None):
        super(GMPool, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)

        self.mab = MAB(dim, dim, dim, num_heads, ln=ln, cluster=cluster, conv=mab_conv)
        
    def forward(self, X, attention_mask=None, graph=None, return_attn=False):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, attention_mask, graph, return_attn)


### GCN convolution for OGB datasets
class GCNConv_OGB(MessagePassing):
    def __init__(self, emb_dim):
        
        super(GCNConv_OGB, self).__init__(aggr='add')

        self.linear_node = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_edge = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

    def forward(self, x, edge_index, edge_attr, edge_weight=None):

        x = self.linear_node(x)
        edge_attr = self.linear_edge(edge_attr)

        row, col = edge_index

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)

        num_nodes = x.size(0)
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

