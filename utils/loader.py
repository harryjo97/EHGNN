import os
import time
from tqdm import tqdm, trange
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


def load_data(args):

    dataset = PygGraphPropPredDataset(name = args.data)
    args.task_type = dataset.task_type

    args.num_node_features = dataset.data.num_node_features
    if args.num_node_features == 0:
        print('No Node Feature')
    
    args.num_edge_features = dataset.data.num_edge_features
    if args.num_edge_features == 0:
        print('No Edge Feature')
    
    args.num_classes = dataset.num_tasks
    args.avg_num_nodes, args.avg_num_edges = np.ceil(np.mean([data.num_nodes for data in dataset])), np.ceil(np.mean([data.num_edges for data in dataset]))

    print('# %s: [Task]-%s [NODE FEATURES]-%d [EDGE FEATURES]-%d [NUM_CLASSES]-%d [AVG_NODES]-%d [AVG_EDGES]-%d' % 
        (dataset, args.task_type, args.num_node_features, args.num_edge_features, args.num_classes, args.avg_num_nodes, args.avg_num_edges))

    return dataset