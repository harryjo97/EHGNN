# Edge Representation Learning with Hypergraphs

Official Code Repository for the paper "Edge Representation Learning with Hypergraphs"

## Abstract

Graph neural networks have recently achieved remarkable success in representing graph-structured data, with rapid progress in both the node embedding and graph pooling methods. Yet, they mostly focus on capturing information from the nodes considering their connectivity, and not much work has been done in representing the edges, which are essential components of a graph. However, for tasks such as graph reconstruction and generation, as well as graph classification tasks for which the edges are important for discrimination, accurately representing edges of a given graph is crucial to the success of the graph representation learning. To this end, we propose a novel edge representation learning framework based on Dual Hypergraph Transformation (DHT), which transforms the edges of a graph into the nodes of a hypergraph. This dual hypergraph construction allows us to apply message passing techniques for node representations to edges. After obtaining edge representations from the hypergraphs, we then cluster or drop edges to obtain holistic graph-level edge representations. We validate our edge representation learning method with hypergraphs on diverse graph datasets for graph representation and generation performance, on which our method largely outperforms existing graph representation learning methods. Moreover, our edge representation learning and pooling method also largely outperforms state-of-the-art graph pooling methods on graph classification, not only because of its accurate edge representation learning, but also due to its lossless compression of the nodes and removal of irrelevant edges for effective message passing.

### Contribution

+ We introduce a novel edge representation learning scheme using Dual Hypergraph Transformation, which exploits the dual hypergraph whose nodes are edges of the original graph, on which we can apply off-the-shelf message-passing schemes designed for node-level representation learning.
+ We propose novel edge pooling methods for graph-level representation learning, namely HyperCluster and HyperDrop, to overcome the limitations of existing node-based pooling methods.
+ We validate our methods on graph reconstruction, generation, and classification tasks, on which they largely outperform existing graph representation learning methods.

## Dependencies

+ Python 3.7.0
+ Pytorch 1.4.0
+ Pytorch Geometric 1.4.3

## Run

+ Graph Classification on TU datasets

```sh
sh ./scripts/classification_TU.sh 0 000
```

+ Graph Classification on OGB datasets

```sh
sh ./scripts/classification_OGB.sh 0 000
```

+ Edge Reconstruction on the ZINC dataset

```sh
sh ./scripts/reconstruction_ZINC.sh 0 000
```

## Citation