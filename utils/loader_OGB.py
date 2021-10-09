import numpy as np
import torch
from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
from models.models import Model_HyperDrop_OGB


def load_data(args):

    dataset = PygGraphPropPredDataset(name = args.data)
    args.task_type = dataset.task_type

    args.num_node_features = dataset.data.num_node_features
    args.num_edge_features = dataset.data.num_edge_features
    
    args.num_classes = dataset.num_tasks
    args.avg_num_nodes = np.ceil(np.mean([data.num_nodes for data in dataset]))
    args.avg_num_edges = np.ceil(np.mean([data.num_edges for data in dataset]))

    print('# %s: [Task]-%s [NODE FEATURES]-%d [EDGE FEATURES]-%d [NUM_CLASSES]-%d [AVG_NODES]-%d [AVG_EDGES]-%d' % 
        (dataset, args.task_type, args.num_node_features, args.num_edge_features, args.num_classes, args.avg_num_nodes, args.avg_num_edges))

    return dataset


def load_dataloader(args, dataset):

    split_idx = dataset.get_idx_split()

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_model(args):

    if args.model == 'HyperDrop':
        model = Model_HyperDrop_OGB(args)

    else:
        raise ValueError("Model Name <{}> is Unknown".format(args.model))

    return model
