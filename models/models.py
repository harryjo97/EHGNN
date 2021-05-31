import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
from torch_geometric.utils import to_dense_batch

from models.layers import HypergraphConv, GCNConv_OGB, GMPool
from torch_geometric.nn import GCNConv
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from math import ceil


class GraphRepresentation(nn.Module):

    def __init__(self, args):

        super(GraphRepresentation, self).__init__()

        self.args = args

        self.num_node_features = args.num_node_features
        self.num_edge_features = args.num_edge_features
        self.nhid = args.num_hidden
        self.num_classes = args.num_classes

        self.edge_ratio = args.edge_ratio
        self.dropout_ratio = args.dropout

        self.enhid = self.nhid

    def DHT(self, edge_index, batch, add_loops=True):

        num_edge = edge_index.size(1)
        device = edge_index.device

        hyperedge_index = edge_index.T.reshape(1,-1)
        hyperedge_index = torch.cat([torch.arange(0,num_edge,1, device=device).repeat_interleave(2).view(1,-1), hyperedge_index], dim=0).long() 

        edge_batch = hyperedge_index[1,:].reshape(-1,2)[:,0]
        edge_batch = torch.index_select(batch, 0, edge_batch)

        if add_loops:
            bincount =  hyperedge_index[1].bincount()
            mask = bincount[hyperedge_index[1]]!=1
            max_edge = hyperedge_index[1].max()
            loops = torch.cat([torch.arange(0,num_edge,1,device=device).view(1,-1), torch.arange(max_edge+1,max_edge+num_edge+1,1,device=device).view(1,-1)], dim=0)

            hyperedge_index = torch.cat([hyperedge_index[:,mask], loops], dim=1)

        return hyperedge_index, edge_batch

    def get_classifier(self, dim=1):

        classifier = nn.Sequential(
            nn.Linear(self.nhid*dim, self.nhid),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid, self.nhid//2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.nhid//2, self.num_classes)
        )

        return classifier

    def get_scoreconvs(self):

        convs = nn.ModuleList()

        for i in range(self.args.num_convs-1):

            conv = HypergraphConv(self.enhid, 1)
            convs.append(conv)

        return convs


class Model_HyperDrop(GraphRepresentation):

    def __init__(self, args):

        super(Model_HyperDrop, self).__init__(args)

        if self.args.data == 'COLLAB':
            self.enhid = self.nhid // 8

        self.convs = self.get_convs()
        self.hyperconvs = self.get_convs(conv_type='Hyper')[:-1]

        self.scoreconvs = self.get_scoreconvs()
        self.classifier = self.get_classifier(dim=3)


    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # edge feature initialization
        if edge_attr is None:
            edge_attr = torch.ones((edge_index.size(1), 1), device=edge_index.device)

        xs = 0

        for _ in range(self.args.num_convs):
            
            hyperedge_index, edge_batch = self.DHT(edge_index, batch)

            if _ == 0:
                x = F.relu( self.convs[_](x, edge_index) )

            else:
                x = F.relu( self.convs[_](x, edge_index, edge_weight) )

            if _ < self.args.num_convs-1:

                edge_attr = F.relu( self.hyperconvs[_](edge_attr, hyperedge_index) )

                score = torch.tanh( self.scoreconvs[_](edge_attr, hyperedge_index).squeeze() )
                perm = topk(score, self.edge_ratio, edge_batch)

                edge_index = edge_index[:,perm]
                edge_attr = edge_attr[perm, :]
                edge_weight = score[perm]
                edge_weight = torch.clamp(edge_weight, min=0, max=1)

            xs += torch.cat([gmp(x,batch), gap(x,batch), gsp(x,batch)], dim=1)

        x = self.classifier(xs)

        return F.log_softmax(x, dim=1)

    def get_convs(self, conv_type='GCN'):

        convs = nn.ModuleList()

        for i in range(self.args.num_convs):

            if conv_type == 'GCN':

                if i == 0 :
                    conv = GCNConv(self.num_node_features, self.nhid)
                else:
                    conv = GCNConv(self.nhid, self.nhid)

            elif conv_type == 'Hyper':

                if i == 0 :
                    conv = HypergraphConv(self.num_edge_features, self.enhid)
                else:
                    conv = HypergraphConv(self.enhid, self.enhid)

            else:
                raise ValueError("Conv Name <{}> is Unknown".format(conv_type))

            convs.append(conv)

        return convs

class Model_HyperDrop_OGB(GraphRepresentation):

    def __init__(self, args):

        super(Model_HyperDrop_OGB, self).__init__(args)

        self.atom_encoder = AtomEncoder(self.nhid)
        self.bond_encoder = BondEncoder(self.nhid)

        self.convs = self.get_convs()
        self.hyperconvs = self.get_convs(conv_type='Hyper')[:-1]

        self.scoreconvs = self.get_scoreconvs()
        self.classifier = self.get_classifier(dim=3)

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)

        xs = 0

        for _ in range(self.args.num_convs):
            
            hyperedge_index, edge_batch = self.DHT(edge_index, batch)

            if _ == 0:
                x = F.relu( self.convs[_](x, edge_index, edge_attr) )

            else:
                x = F.relu( self.convs[_](x, edge_index, edge_attr, edge_weight) )

            if _ < self.args.num_convs-1:

                edge_attr = F.relu( self.hyperconvs[_](edge_attr, hyperedge_index) )

                score = torch.tanh( self.scoreconvs[_](edge_attr, hyperedge_index).squeeze() )
                perm = topk(score, self.edge_ratio, edge_batch)

                edge_index = edge_index[:,perm]
                edge_attr = edge_attr[perm, :]
                edge_weight = score[perm]
                edge_weight = torch.clamp(edge_weight, min=0, max=1)

            xs += torch.cat([gmp(x,batch), gap(x,batch), gsp(x,batch)], dim=1)

        x = self.classifier(xs)

        return x

    def get_convs(self, conv_type='GCN_OGB'):

        convs = nn.ModuleList()

        for i in range(self.args.num_convs):

            if conv_type == 'GCN_OGB':

                conv = GCNConv_OGB(self.nhid)

            elif conv_type == 'Hyper':

                conv = HypergraphConv(self.enhid, self.enhid)

            else:
                raise ValueError("Conv Name <{}> is Unknown".format(conv_type))

            convs.append(conv)

        return convs
 
class Model_HyperCluster(GraphRepresentation):

    def __init__(self, args):

        super(Model_HyperCluster, self).__init__(args)

        self.pooling_ratio = self.args.edge_ratio
        self.num_seeds_edge = ceil(self.pooling_ratio * self.args.avg_num_edges)
        self.ln = self.args.ln
        self.num_heads = self.args.num_heads
        self.cluster = self.args.cluster

        self.convs = self.get_convs()
        self.unconvs = self.get_unconvs()
        self.last = HypergraphConv(self.nhid, self.num_edge_features)

        self.pool = GMPool(self.nhid, self.num_heads, self.num_seeds_edge, ln=self.ln, cluster=self.cluster, mab_conv='Hyper')

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        hyperedge_index, edge_batch = self.DHT(edge_index, batch)

        # MPs
        for _ in range(self.args.num_convs):
            edge_attr = F.relu( self.convs[_](edge_attr, hyperedge_index) )

        # Pool
        batch_edge_attr, mask = to_dense_batch(edge_attr, edge_batch)

        extended_attention_mask = mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

        batch_edge_attr, attn = self.pool(batch_edge_attr, attention_mask=extended_attention_mask, graph=(edge_attr, hyperedge_index, edge_batch), return_attn=True)

        # Unpool 
        edge_attr = torch.bmm(attn.transpose(1, 2), batch_edge_attr)   
        edge_attr = edge_attr[mask]

        # MPs
        for _ in range(self.args.num_convs):
            edge_attr = F.relu( self.unconvs[_](edge_attr, hyperedge_index) )

        edge_attr = self.last(edge_attr, hyperedge_index)

        return edge_attr

    def get_convs(self):

        convs = nn.ModuleList()

        for i in range(self.args.num_convs):

            if i == 0:
                conv = HypergraphConv(self.num_edge_features, self.nhid)
            else:
                conv = HypergraphConv(self.nhid, self.nhid)

            convs.append(conv)

        return convs

    def get_unconvs(self):

        unconvs = nn.ModuleList()

        for i in range(self.args.num_convs):

            conv = HypergraphConv(self.nhid, self.nhid)

            unconvs.append(conv)

        return unconvs

