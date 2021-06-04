# Edge Representation Learning with Hypergraphs

This code is the official implementation of Edge Representation Learning with Hypergraphs.

## Dependencies

+ Python 3.7.0
+ Pytorch 1.4.0
+ Pytorch Geometric 1.4.3

## Training and Evaluation

We provide the commands for the following tasks: Graph Reconstruction and Graph Classification

For each command, the first argument denotes the gpu id and the second argument denotes the experiment number.

+ Edge Reconstruction on the ZINC dataset

```sh
sh ./scripts/reconstruction_ZINC.sh 0 000
```

+ Graph Classification on TU datasets

```sh
sh ./scripts/classification_TU.sh 0 000
```

+ Graph Classification on OGB datasets

```sh
sh ./scripts/classification_OGB.sh 0 000
```

## Results

Please see our paper for the results.