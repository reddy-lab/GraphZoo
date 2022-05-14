"""Input parameters for the library"""
import argparse
from graphzoo.utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (0.05, 'initial learning rate (type: float)'),
        'dropout': (0.0, 'dropout probability (type: float)'),
        'cuda': (-1, 'which cuda device to use or -1 for cpu training (type: int)'),
        'repeat': (10, 'number of times to repeat the experiment (type: int)'),
        'optimizer': ('Adam', 'which optimizer to use, can be any of [Adam, RiemannianAdam, RiemannianSGD] (type: str)'),
        'epochs': (5000, 'maximum number of epochs to train for (type:int)'),
        'weight-decay': (0.0, 'l2 regularization strength (type: float)'),
        'momentum': (0.999, 'momentum in optimizer (type: float)'),
        'patience': (100, 'patience for early stopping (type: int)'),
        'seed': (1234, 'seed for training (type: int)'),
        'log-freq': (5, 'how often to compute print train/val metrics in epochs (type: int)'),
        'eval-freq': (1, 'how often to compute val metrics in epochs (type: int)'),
        'save': (0, '1 to save model and logs and 0 otherwise (type: int)'),
        'save-dir': (None, 'path to save training logs and model weights (type: str)'),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant (type: int)'),
        'gamma': (0.5, 'gamma for lr scheduler (type: float)'),
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping (type: float)'),
        'min-epochs': (100, 'do not early stop before min-epochs (type: int)'),
        'betas': ((0.9, 0.999), 'coefficients used for computing running averages of gradient and its square (type: Tuple[float, float])'),
        'eps': (1e-8, 'term added to the denominator to improve numerical stability (type: float)'),
        'amsgrad': (False, 'whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence of Adam and Beyond` (type: bool)'),
        'stabilize': (None, 'stabilize parameters if they are off-manifold due to numerical reasons every ``stabilize`` steps (type: int)'),
        'dampening': (0,'dampening for momentum (type: float)'),
        'nesterov': (False,'enables Nesterov momentum (type: bool)')
    },
    'model_config': {
        'task': ('nc', 'which tasks to train on, can be any of [lp, nc] (type: str)'),
        'model': ('HGCN', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN,HGAT] (type: str)'),
        'dim': (128, 'embedding dimension (type: int)'),
        'manifold': ('PoincareBall', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall] (type: str)'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature (type: float)'),
        'r': (2.0, 'fermi-dirac decoder parameter for lp (type: float)'),
        't': (1.0, 'fermi-dirac decoder parameter for lp (type: float)'),
        'pretrained-embeddings': (None, 'path to pretrained embeddings (.npy file) for Shallow node classification (type: str)'),
        'num-layers': (2, 'number of hidden layers in encoder (type: int)'),
        'bias': (1, 'whether to use bias (1) or not (0) (type: int)'),
        'act': ('relu', 'which activation function to use or None for no activation (type: str)'),
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim (type: int)'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks (type: float)'),
        'use-att': (0, 'whether to use hyperbolic attention (1) or not (0) (type: int)'),
        'local-agg': (0, 'whether to local tangent space aggregation (1) or not (0) (type: int)')
    },
    'data_config': {
        'dataset': ('cora', 'which dataset to use, can be any of [cora, pubmed, airport, disease_nc, disease_lp, ppi, citeseer, webkb] (type: str)'),
        'datapath': (None, 'path to raw data (type: str)'),
        'val-prop': (0.05, 'proportion of validation edges for link prediction (type:float)'),
        'test-prop': (0.1, 'proportion of test edges for link prediction (type: float)'),
        'use-feats': (1, 'whether to use node features (1) or not (0 in case of Shallow methods) (type: int)'),
        'normalize-feats': (1, 'whether to normalize input node features (1) or not (0) (type: int)'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix (1) or not(0) (type: int)'),
        'split-seed': (1234, 'seed for data splits (train/test/val) (type: int)'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
