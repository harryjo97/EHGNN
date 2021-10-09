import torch
from torch_geometric.data import DataLoader
import numpy as np
from functools import reduce

from utils.data import get_dataset
from models.models import Model_HyperDrop


def load_data(args):

    dataset = get_dataset(args.data, normalize=args.normalize)
    
    args.num_node_features = dataset.data.num_node_features
    if args.num_node_features == 0:
        print('Added Node Feature')
        args.num_node_features = dataset.num_features
    
    args.num_edge_features = dataset.data.num_edge_features
    if args.num_edge_features == 0:
        print('Added Edge Feature')
        args.num_edge_features = 1

    args.num_classes = dataset.num_classes
    args.avg_num_nodes = np.ceil(np.mean([data.num_nodes for data in dataset]))
    args.avg_num_edges = np.ceil(np.mean([data.num_edges for data in dataset]))

    print(f'# {dataset}: [NODE FEATURES]-{args.num_node_features} [EDGE FEATURES]-{args.num_edge_features} '
            f'[NUM_CLASSES]-{args.num_classes} [AVG_NODES]-{args.avg_num_nodes} [AVG_EDGES]-{args.avg_num_edges}')

    print(f"Nodes : {np.mean([data.num_nodes for data in dataset]):.2f}   " 
            f"Edges : {np.mean([data.num_edges for data in dataset])/2:.2f}")

    return dataset


def load_dataloader(args, dataset, fold_number, val_fold_number):

    train_idxes = torch.as_tensor(np.loadtxt(f'./datasets/{args.data}/10fold_idx/train_idx-{fold_number}.txt',
                                            dtype=np.int32), dtype=torch.long)
    val_idxes = torch.as_tensor(np.loadtxt(f'./datasets/{args.data}/10fold_idx/test_idx-{val_fold_number}.txt',
                                            dtype=np.int32), dtype=torch.long)     
    test_idxes = torch.as_tensor(np.loadtxt(f'./datasets/{args.data}/10fold_idx/test_idx-{fold_number}.txt',
                                            dtype=np.int32), dtype=torch.long)

    all_idxes = reduce(np.union1d, (train_idxes, val_idxes, test_idxes))
    assert len(all_idxes) == len(dataset)

    train_idxes = torch.as_tensor(np.setdiff1d(train_idxes, val_idxes))

    train_set, val_set, test_set = dataset[train_idxes], dataset[val_idxes], dataset[test_idxes]

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_model(args):

    if args.model == 'HyperDrop':
        model = Model_HyperDrop(args)

    else:
        raise ValueError("Model Name <{}> is Unknown".format(args.model))

    return model
