import torch
from torch_geometric.data import DataLoader

from utils.molecule_utils import ZINC, dataset_statistic_node, dataset_statistic_edge, mol_from_graphs, to_one_hot
from models.models import Model_HyperCluster


def load_data(args):
        
    subset = True
    
    train_dataset = ZINC("./zinc", subset=subset, split='train', transform=to_one_hot)
    valid_dataset = ZINC("./zinc", subset=subset, split='val', transform=to_one_hot)
    test_dataset = ZINC("./zinc", subset=subset, split="test", transform=to_one_hot)
    
    args.num_node_features = 28 
    args.num_edge_features = 5
    args.num_classes = 5
    args.avg_num_nodes = dataset_statistic_node(train_dataset)
    args.avg_num_edges = dataset_statistic_edge(train_dataset)

    return train_dataset, valid_dataset, test_dataset


def load_dataloader(args):

    train_dataset, val_dataset, test_dataset = load_data(args)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


def load_model(args):

    if args.model == 'HyperCluster':
        model = Model_HyperCluster(args)

    else:
        raise ValueError("Model Name <{}> is Unknown".format(args.model))

    return model

