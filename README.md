# GraphZoo
> PyTorch version of [GraphZoo](https://github.com/reddy-lab/GraphZoo).

> Facilitating learning, using, and designing graph processing pipelines/models systematically.

We present a novel framework GraphZoo, that makes learning, using, and designing graph processing pipelines/models systematic by abstraction over the redundant components. The framework contains a powerful library that supports several hyperbolic manifolds and an easy-to-use modular framework to perform graph processing tasks which aids researchers in different components, namely, (i) reproduce evaluation pipelines of state-of-the-art approaches, (ii) design new hyperbolic or Euclidean graph networks and compare them against the state-of-the art approaches on standard benchmarks, (iii) add custom datasets for evaluation, (iv) add new tasks and evaluation criteria. 

## Installation

### Using Github source:
```
git clone https://github.com/reddy-lab/GraphZoo.git
cd GraphZoo
python setup.py install
```

### Using Pypi (under development, install from source):

```
pip install graphzoo
```

## Getting Started in 60 Seconds

To train a Hyperbolic Graph Convolutional Networks model for node classification task on Cora dataset, make use of GraphZoo APIs customized loss functions and evaluation metrics for this task.

Prepare input data:

```python
import graphzoo as gz
import torch
from graphzoo.config import parser

params = parser.parse_args(args=[])
params.dataset='cora'
params.datapath='data/cora'
data = gz.dataloader.DataLoader(params)
```

Initialize the model and fine-tune the hyperparameters:

```python
params.task='nc'
params.model='HGCN'
params.manifold='PoincareBall'
params.dim=128
model= gz.models.NCModel(params)
```

`Trainer` is used to control the training flow:

```python
optimizer = gz.optimizers.RiemannianAdam(params=model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
trainer=gz.trainers.Trainer(params,model,optimizer,data)
trainer.run()
trainer.evaluate()
```
## Getting Started Using Command Line
To train a Hyperbolic Graph Convolutional Networks model for node classification task on Cora dataset using command line:

```python
cd GraphZoo
python graphzoo/trainers/train.py --task nc --dataset cora --datapath <your datapath> --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall --log-freq 5 --cuda 0 --c None
```
## Customizing Input Arguments

Various flags can be modified in the `graphzoo.config` module by the user.

### DataLoader

```python
     """
    GraphZoo Dataloader

    Input Parameters
    ----------
        'dataset': ('cora', 'which dataset to use, can be any of [cora, pubmed, airport, disease_nc, disease_lp] (type: str)')
        'datapath': (None, 'path to raw data (type: str)')
        'val-prop': (0.05, 'proportion of validation edges for link prediction (type:float)')
        'test-prop': (0.1, 'proportion of test edges for link prediction (type: float)')
        'use-feats': (1, 'whether to use node features (1) or not (0 in case of Shallow methods) (type: int)')
        'normalize-feats': (1, 'whether to normalize input node features (1) or not (0) (type: int)')
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix (1) or not(0) (type: int)')
        'split-seed': (1234, 'seed for data splits (train/test/val) (type: int)')

    API Input Parameters
    ----------
        args: list of above defined input parameters from `graphzoo.config`
    
    """

```

### Models

```python
    """
    Base model for graph embedding tasks

    Input Parameters
    ----------
        'task': ('nc', 'which tasks to train on, can be any of [lp, nc] (type: str)')
        'model': ('HGCN', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN,HGAT] (type: str)')
        'dim': (128, 'embedding dimension (type: int)')
        'manifold': ('PoincareBall', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall] (type: str)')
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature (type: float)')
        'r': (2.0, 'fermi-dirac decoder parameter for lp (type: float)')
        't': (1.0, 'fermi-dirac decoder parameter for lp (type: float)')
        'pretrained-embeddings': (None, 'path to pretrained embeddings (.npy file) for Shallow node classification (type: str)')
        'num-layers': (2, 'number of hidden layers in encoder (type: int)')
        'bias': (1, 'whether to use bias (1) or not (0) (type: int)')
        'act': ('relu', 'which activation function to use or None for no activation (type: str)')
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim (type: int)')
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks (type: float)')
        'use-att': (0, 'whether to use hyperbolic attention (1) or not (0) (type: int)')
        'local-agg': (0, 'whether to local tangent space aggregation (1) or not (0) (type: int)')
        'n_classes': (7, 'number of classes in the dataset (type: int)')
        'n_nodes': (2708, 'number of nodes in the graph (type: int)')
        'feat_dim': (1433, 'feature dimension of the dataset (type: int)') 
        
    API Input Parameters
    ----------
        args: list of above defined input parameters from `graphzoo.config`
    """
```

### Trainer

```python
    """
    GraphZoo Trainer

    Input Parameters
    ----------
        'lr': (0.01, 'initial learning rate (type: float)')
        'dropout': (0.5, 'dropout probability (type: float)')
        'cuda': (-1, 'which cuda device to use or -1 for cpu training (type: int)')
        'device': ('cpu', 'which device to use cuda:$devicenumber for GPU or cpu for CPU (type: str)')
        'repeat': (10, 'number of times to repeat the experiment (type: int)')
        'optimizer': ('Adam', 'which optimizer to use, can be any of [Adam, RiemannianAdam, RiemannianSGD] (type: str)')
        'epochs': (5000, 'maximum number of epochs to train for (type:int)')
        'weight-decay': (0.001, 'l2 regularization strength (type: float)')
        'momentum': (0.999, 'momentum in optimizer (type: float)')
        'patience': (100, 'patience for early stopping (type: int)')
        'seed': (1234, 'seed for training (type: int)')
        'log-freq': (5, 'how often to compute print train/val metrics in epochs (type: int)')
        'eval-freq': (1, 'how often to compute val metrics in epochs (type: int)')
        'save': (0, '1 to save model and logs and 0 otherwise (type: int)')
        'save-dir': (None, 'path to save training logs and model weights (type: str)')
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant (type: int)')
        'gamma': (0.5, 'gamma for lr scheduler (type: float)')
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping (type: float)')
        'min-epochs': (100, 'do not early stop before min-epochs (type: int)')
        'betas': ((0.9, 0.999), 'coefficients used for computing running averages of gradient and its square (type: Tuple[float, float])')
        'eps': (1e-8, 'term added to the denominator to improve numerical stability (type: float)')
        'amsgrad': (False, 'whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence of Adam and Beyond` (type: bool)')
        'stabilize': (None, 'stabilize parameters if they are off-manifold due to numerical reasons every ``stabilize`` steps (type: int)')
        'dampening': (0,'dampening for momentum (type: float)')
        'nesterov': (False,'enables Nesterov momentum (type: bool)')

    API Input Parameters
    ----------
        args: list of above defined input parameters from `graphzoo.config`
        optimizer: a :class:`optim.Optimizer` instance
        model: a :class:`BaseModel` instance
    
    """
```

## Customizing the Framework

### Adding Custom Dataset

1. Add the dataset files in the `data` folder of the source code.
2. To run this code on new datasets, please add corresponding data processing and loading in `load_data_nc` and `load_data_lp` functions in `dataloader/dataloader.py` in the source code.

Output format for node classification dataloader is:

```
data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
```
Output format for link prediction dataloader is:

```
data = {'adj_train': adj_train, 'features': features, ‘train_edges’: train_edges, ‘train_edges_false’: train_edges_false,  ‘val_edges’: val_edges, ‘val_edges_false’: val_edges_false, ‘test_edges’: test_edges, ‘test_edges_false’: test_edges_false, 'adj_train_norm':adj_train_norm}
```

### Adding Custom Layers

1. Attention layers can be added in `layers/att_layers.py` in the source code by adding a class in the file.
2. Hyperbolic layers can be added in `layers/hyp_layers.py` in the source code by adding a class in the file.
3. Other layers like a single GCN layer can be added in `layers/layers.py` in the source code by adding a class in the file.

### Adding Custom Models

1. After adding custom layers, custom models can be added in `models/encoders.py` in the source code by adding a class in the file.
2. After adding custom layers, custom decoders to calculate the final output can be added in `models/decoders.py` in the source code by adding a class in the file. Default decoder is the `LinearDecoder`.

## Datasets 

The included datasets are the following and they need to be downloaded from the [link](https://data.world/reddy-lab/graphzoo):
1. Cora
2. Pubmed
3. Disease
4. Airport

## Models

### Shallow Methods 
1. Shallow Euclidean 
2. Shallow Hyperbolic

### Neural Network Methods
1. Multi-Layer Perceptron (MLP)
2. Hyperbolic Neural Networks (HNN) 

### Graph Neural Network Methods
1. Graph Convolutional Neural Networks (GCN) 
2. Graph Attention Networks (GAT)
3. Hyperbolic Graph Convolutions (HGCN) 
4. Hyperbolic Graph Attention Networks (HGAT)


## Package References

[Tutorials](https://github.com/reddy-lab/GraphZoo/tree/main/tutorials) (jupyter notebooks under development)

Documentation (under develpoment)

## Code References

Some of the code was forked from the following repositories.
- [hgcn](https://github.com/HazyResearch/hgcn)
- [hgat](https://github.com/oom-debugger/hyperbolic-layers/tree/8ead8b713fee28f830dd8b33a1468082e0eeae50/py_hnn)
- [geoopt](https://github.com/geoopt/geoopt)
- [pygcn](https://github.com/tkipf/pygcn/tree/master/pygcn)
- [gae](https://github.com/tkipf/gae/tree/master/gae)
- [hyperbolic-image-embeddings](https://github.com/leymir/hyperbolic-image-embeddings)
- [pyGAT](https://github.com/Diego999/pyGAT)
- [poincare-embeddings](https://github.com/facebookresearch/poincare-embeddings)

## Model References

## Citation
If you use GraphZooZoo in your research, please use the following BibTex entry.

```
@inproceedings{10.1145/3487553.3524241,
author = {Vyas, Anoushka and Choudhary, Nurendra and Khatir, Mehrdad and Reddy, Chandan K.},
title = {GraphZoo: A Development Toolkit for Graph Neural Networks with Hyperbolic Geometries},
year = {2022},
isbn = {9781450391306},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3487553.3524241},
doi = {10.1145/3487553.3524241},
booktitle = {Companion Proceedings of the Web Conference 2022},
keywords = {graph learning, graph neural network, hyperbolic models, software},
location = {Lyon, France},
series = {WWW '22}
}
```


## License
[GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.en.html)

Copyright (c) 2022 Dr. Reddy's lab, Department of Computer Science, Virginia Tech
