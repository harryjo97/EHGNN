# Edge Representation Learning with Hypergraphs

Official Code Repository for the paper "Edge Representation Learning with Hypergraphs" (NeurIPS 2021): https://arxiv.org/abs/2106.15845.

In this repository, we implement the *Dual Hypergraph Transformation* (DHT) and two edge pooling methods *HyperDrop* and *HyperCluster*.


<p align="center">
    <img width="750" src="assets/model_overview.jpg"/>
</p>

### Contribution

+ We introduce a novel edge representation learning scheme using Dual Hypergraph Transformation, which exploits the dual hypergraph whose nodes are edges of the original graph, on which we can apply off-the-shelf message-passing schemes designed for node-level representation learning.
+ We propose novel edge pooling methods for graph-level representation learning, namely HyperCluster and HyperDrop, to overcome the limitations of existing node-based pooling methods.
+ We validate our methods on graph reconstruction, generation, and classification tasks, on which they largely outperform existing graph representation learning methods.


## Dependencies

EHGNN is built in **Python 3.7.0** and **Pytorch 1.4.0**. Please use the following command to install the requirements:

```sh
pip install -r requirements.txt
```

additionally run the following command:

```sh
conda install -c conda-forge ogb=1.3.0
conda install -c huggingface transformers=4.4.2
conda install -c conda-forge rdkit=2020.03.3.0
```

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

## Citation

If you found the provided code with our paper useful in your work, we kindly request that you cite our work.

```BibTex
@inproceedings{jo2021ehgnn,
  author    = {Jaehyeong Jo and
               Jinheon Baek and
               Seul Lee and
               Dongki Kim and
               Minki Kang and
               Sung Ju Hwang},
  title     = {Edge Representation Learning with Hypergraphs},
  booktitle = {Advances in Neural Information Processing Systems 34: Annual Conference
               on Neural Information Processing Systems 2021, NeurIPS 2021, December
               6-14, 2021, virtual},
  pages     = {7534--7546},
  year      = {2021}
}
```