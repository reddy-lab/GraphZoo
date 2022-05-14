"""
Base model class
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphzoo.layers.layers import FermiDiracDecoder
from graphzoo import manifolds
import graphzoo.models.encoders as encoders
from graphzoo.models.decoders import model2decoder
from graphzoo.utils.eval_utils import acc_f1


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks

    Input Parameters
    ----------
        'task': ('nc', 'which tasks to train on, can be any of [lp, nc] (type: str)')
        'model': ('HGCN', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN, HGAT] (type: str)')
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

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
            
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)

        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
       
        self.weights = torch.Tensor([1.] * args.n_classes)
        
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class LPModel(BaseModel):
    """
    Base model for link prediction task
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)

    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.dc.forward(sqdist)
        return probs

    def compute_metrics(self, embeddings, data, split):
        if split == 'train':
            nb_false_edges = len(data['train_edges_false'])
            nb_edges = len(data['train_edges'])
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, nb_false_edges, nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

