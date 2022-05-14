"""Graph decoders"""
from graphzoo import manifolds
import torch.nn as nn
import torch.nn.functional as F
from graphzoo.layers.att_layers import GraphAttentionLayer
from graphzoo.layers.layers import GraphConvolution, Linear
from graphzoo.layers.hyp_att_layers import GraphAttentionLayer as HGraphAttentionLayer

class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs


class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder
    """

    def __init__(self, c, args):
        super(GCNDecoder, self).__init__(c)
        act = lambda x: x
        self.cls = GraphConvolution(args.dim, args.n_classes, args.dropout, act, args.bias)
        self.decode_adj = True


class GATDecoder(Decoder):
    """
    Graph Attention Decoder
    """

    def __init__(self, c, args):
        super(GATDecoder, self).__init__(c)
        self.cls = GraphAttentionLayer(args.dim, args.n_classes, args.dropout, F.elu, args.alpha, 1, True)
        self.decode_adj = True


class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models
    """

    def __init__(self, c, args):
        super(LinearDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.cls = Linear(self.input_dim, self.output_dim, args.dropout, lambda x: x, self.bias)
        self.decode_adj = False

    def decode(self, x, adj):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        return super(LinearDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )

class HGATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(HGATDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.cls = HGraphAttentionLayer(
                manifold=self.manifold,
                input_dim=args.dim, 
                output_dim=args.n_classes, 
                dropout=args.dropout, 
                activation=F.elu, 
                alpha=args.alpha, 
                nheads=1, 
                concat=True,  
                curvature=self.c, 
                use_bias= args.bias)
        self.decode_adj = True

    def decode(self, x, adj):
        x = super(HGATDecoder, self).decode(x, adj)
        return self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)

model2decoder = {
    'GCN': GCNDecoder,
    'GAT': GATDecoder,
    'HNN': LinearDecoder,
    'HGCN': LinearDecoder,
    'HGAT': HGATDecoder,
    'MLP': LinearDecoder,
    'Shallow': LinearDecoder
}

