"""Hyperbolic Graph Attention layers"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from graphzoo.layers.hyp_layers import HypLinear, HypAct
from graphzoo.manifolds.poincare import PoincareBall
from graphzoo.manifolds.hyperboloid import Hyperboloid
from graphzoo.manifolds.euclidean import Euclidean
from graphzoo.layers.att_layers import SpecialSpmm

class AdjustableModule(nn.Module):

    def __init__(self, curvature):
        super(AdjustableModule, self).__init__()
        self.curvature = curvature
    
    def update_curvature(self, curvature):
        self.curvature = curvature


class SpGraphAttentionLayer(AdjustableModule):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, manifold, in_features, out_features, dropout, alpha, activation, curvature = 1, use_bias=False):
        super(SpGraphAttentionLayer, self).__init__(curvature)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.use_bias = use_bias
        self.manifold  = manifold
        self.linear = HypLinear(manifold, in_features, out_features, self.curvature, dropout, use_bias=use_bias)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        self.act = activation

    def update_curvature(self, c):
        super(SpGraphAttentionLayer, self).update_curvature(c)
        self.linear.update_curvature(c)

    def forward(self, input, adj):
        N = input.size()[0]
        edge = adj._indices()
        h = self.linear(input)

        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = -self.manifold.sqdist(h[edge[0, :], :], h[edge[1, :], :], c=self.curvature).unsqueeze(0)
        
        ########################Euclidean Block (START)########################
        # convert h to Euclidean space.
        h = self.manifold.proj_tan0(self.manifold.logmap0(h, c=self.curvature), c=self.curvature)

        edge_e = torch.exp(-self.leakyrelu(edge_h.squeeze()))
        
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        ones = torch.ones(size=(N, 1))
        if h.is_cuda:
            ones = ones.cuda()
        e_rowsum = self.special_spmm(indices=edge, values=edge_e, shape=torch.Size([N, N]), b=ones)
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(indices=edge, values=edge_e, shape=torch.Size([N, N]), b=h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = self.act(h_prime)
        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature), c=self.curvature)
        return out
     
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer(AdjustableModule):
    def __init__(self, manifold, input_dim, output_dim, dropout, activation, alpha, nheads, concat, curvature, use_bias):
        """Sparse version of GAT."""
        super(GraphAttentionLayer, self).__init__(curvature)
        self.dropout = dropout
        self.output_dim = output_dim
        self.nheads = nheads
        self.manifold = manifold
        if self.nheads > 1 and isinstance(manifold, Hyperboloid):
            tmp_dim = (self.output_dim-1) * nheads + 1
            self.linear_out = HypLinear(manifold, tmp_dim, self.nheads*self.output_dim, self.curvature, dropout=0, use_bias=False)

        self.attentions = [SpGraphAttentionLayer(
                manifold,
                input_dim,
                output_dim,
                dropout=dropout,
                alpha=alpha,
                activation=activation,
                curvature=curvature,
                use_bias=use_bias) for _ in range(nheads)]
        self.concat = concat
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def update_curvature(self, c):
        super(GraphAttentionLayer, self).update_curvature(c)
        for layer in self.attentions:
            layer.update_curvature(c)

    def forward(self, input):
        x, adj = input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        x = F.dropout(x, self.dropout, training=self.training)
        if self.concat:
            if isinstance(self.manifold, Euclidean):
                h = torch.cat([att(x, adj) for att in self.attentions], dim=1)
            elif self.nheads > 1:
                h = torch.stack([att(x, adj) for att in self.attentions], dim=-2)
                h = self.manifold.concat(h, c=self.curvature).squeeze()
                if isinstance(self.manifold, Hyperboloid):
                    h = self.linear_out(h)
            else:  # No concat
                h = self.attentions[0](x, adj)
        else:
            raise ValueError('aggregation is not supported ')
        h = F.dropout(h, self.dropout, training=self.training)
        return (h, adj)

class SharedSelfAttention(Module):
    """
    Hyperbolic attention layer with self-attention matrix.
    """
    def __init__(self, manifold, input_dim, output_dim, curvature, activation=None, alpha=0.2, dropout=0.1, use_bias=True):
        super(SharedSelfAttention, self).__init__()
        self.curvature = curvature
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.manifold = manifold

        # As the first step, we create an attention matrix using a linear layer
        # followed by a leakyReLU. 
        # inspired from "Graph Attention Networks" by P. Veickovic ICLR 2018.

        # Note: the paper uses shared attention matrix, which means it is the same W
        # for all inputs of all nodes. W_dim(in_dim=d_model, out_dim=d_k)        
        # However, if we want to have node-specific attention then
        # W_dim(graph_nodes * d_model, graph_nodes * graph_nodes * self.d_k)
        
        self.att_input_linear = HypLinear(
                manifold=self.manifold,
                in_features=self.input_dim,
                out_features=self.output_dim, 
                c=self.curvature, 
                dropout=dropout, 
                use_bias=use_bias)   
        nn.init.xavier_uniform(self.att_input_linear.weight)

        self.hyp_act = None
        if activation:
            self.hyp_act = HypAct(
                    manifold=self.manifold, 
                    c_in=self.curvature,
                    c_out=self.curvature, 
                    act=activation)


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'         

class SharedSelfAttentionV0(SharedSelfAttention):
    """
    Hyperbolic attention layer with self-attention matrix.
    
    Uses mobius midpoint for calculating attention coefficient.
    """
    def __init__(self, manifold, input_dim, output_dim, curvature, activation=None, alpha=0.2, dropout=0.1, use_bias=True):
        super(SharedSelfAttentionV0, self).__init__(manifold, input_dim, 
             output_dim, curvature, activation, alpha, dropout, use_bias)
       
    def forward(self, hyp_features, edges):
        if torch.any(torch.isnan(hyp_features)):
            raise ValueError('input to SharedSelfAttentionV0 has NaaN values')
        att_per_node = self.att_input_linear(hyp_features)
        # Alternatively you cann pass hyp_features from a seperate linear layer
        # to create reduced_hyp_features
        reduced_hyp_features = att_per_node

        # create adjaceny matrix from edge info
        mask = edges.to_dense().transpose(0,1)
        if torch.nonzero(mask).numel() == 0:
            raise ValueError('adjacency matrix must have at least 1 edge.')
        hyp_att_embeddings = []

        for src_node, incoming_edges in enumerate(mask):
            # calculate the activation for each node
            masked_v = []
            masked_a = []
            for tgt_node, val in enumerate(incoming_edges):
                if val > 0.01:
                    # we define attention coefficient with the following formula.
                    coef = -1 * val * Hyperboloid().sqdist(att_per_node[tgt_node], att_per_node[src_node], c=self.curvature)
                    if torch.isnan(coef):
                        raise ValueError('we cannot have attentions coeficinet as NaaN')
                    masked_a.append(coef)
                    masked_v.append(reduced_hyp_features[tgt_node])
            if not masked_a and not masked_v:
                raise ValueError(
                        'A graph node must have at least one incoming edge.')
        
            masked_a = torch.FloatTensor(torch.stack(masked_a).squeeze(-1))
            masked_v = torch.stack(masked_v)
            # Note since for attention matrix we use linear layer which includes 
            # droupout rate as well. we omit the separate drop out layer.
            # project the hyperbolic vector to poincare model.
            poincare_v = PoincareBall().from_hyperboloid(x=masked_v, c=self.curvature)
            # calculate attention embeddings for each node.          
            att_embed = PoincareBall().mobius_midpoint(a=masked_a, v=poincare_v)  
            hyp_att_em = Hyperboloid().from_poincare(att_embed, c=self.curvature)
            hyp_att_embeddings.append(hyp_att_em)
        
        hyp_att_embeddings = torch.stack(hyp_att_embeddings)   
        
        if self.hyp_act:
            hyp_att_embeddings = self.hyp_act(hyp_att_embeddings)    
            return hyp_att_embeddings


class MultiHeadGraphAttentionLayer(Module):

    def __init__(self, manifold, input_dim, output_dim, dropout, curvature=1, activation=None, alpha=0.2, nheads=1, concat=None, self_attention_version='v0'):
        """Sparse version of GAT."""
        super(MultiHeadGraphAttentionLayer, self).__init__()
        if self_attention_version == 'v0':
            self_attention_layer_class = SharedSelfAttentionV0
        else:
            raise ValueError('Unknown self-attention version!')
        self.manifold = manifold
        self.dropout = dropout
        self.output_dim = output_dim
        self.curvature = curvature
        self.manifold = Hyperboloid
        self.attentions = [self_attention_layer_class(
                manifold=self.manifold,
                input_dim=input_dim, 
                output_dim=self.output_dim, 
                curvature=self.curvature, 
                alpha=alpha,
                activation=activation,
                dropout=self.dropout, 
                use_bias=False) for _ in range(nheads)]
        self.linear_out = None
        if nheads > 1:
            self.linear_out = HypLinear(
                manifold=self.manifold,
                in_features=nheads * (self.output_dim - 1) + 1,
                out_features=nheads * self.output_dim,
                c=self.curvature, 
                dropout=0.0, 
                use_bias=False)

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, input):
        x, adj = input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        x = F.dropout(x, self.dropout, training=self.training)
        p_h = PoincareBall().from_hyperboloid(torch.stack([att(x, adj) for att in self.attentions], dim=1))
        p_h = PoincareBall().concat(p_h)
        h = Hyperboloid().from_poincare(p_h)

        if self.linear_out:
            h = self.linear_out(h)    

        h = F.dropout(h, self.dropout, training=self.training)
        return (h, adj)

class HypGraphSelfAttentionLayer(Module):
    """
    Hyperbolic attention layer with node-specific attention matrix.
    """
    def __init__(self, graph_size, vector_dim, curvature, dropout=0.1, use_bias=False):
        super(HypGraphSelfAttentionLayer, self).__init__()
        self.curvature = curvature
        self.vector_dim = vector_dim
        self.graph_dim = graph_size
        # As the first step, we create an attention matrix using a linear layer
        # followed by a leakyReLU. 
        # inspired from "Graph Attention Networks" by P. Veickovic ICLR 2018.
        self.input_linear = HypLinear(
                manifold=PoincareBall,
                in_features=self.graph_dim * self.vector_dim, 
                out_features=self.graph_dim * self.graph_dim * self.vector_dim, 
                c=self.curvature, 
                dropout=dropout, 
                use_bias=use_bias)

        self.att_out_linear = HypLinear(
                manifold=PoincareBall,
                in_features=self.graph_dim * self.vector_dim,
                out_features=self.graph_dim * self.vector_dim,
                c=self.curvature,
                dropout=False,
                use_bias=use_bias)

    @classmethod
    def graph_attention(self, a, v, mask):
        """calculare the graph attention for a single node.
        
        Note: it is based on the eq.8, eq.9 in "Hyperbolic graph attention network" paper.
        
        args:
            a: attention coefficient vector of dim(M,M,N).
            v: M*N dimensional matrix, where M is the number of 
                vectors of dim N. 
            mask: a vector of dim (M, M) that indicates the connection map of
                the nodes in the graph.
        returns:
            a vector of dim(M,N) corresponding to the attention embeddings for
            the given node.
        """

        masked_v = []
        masked_a = []
        h = []
        for i, _ in enumerate(v):
            a_i = a[i].view()
            mask_i = mask[i].view()
            # For each node we extract the nodes in the connection map, then 
            # we calculate the mid-point for that node. This needs to be
            # repeated for all nodes in the graph.
            for idx, mask_bit in enumerate(mask_i):
                if mask_bit == 1:
                    masked_v.append(v[idx])
                    masked_a.append(a_i[idx])
            h.append(PoincareBall()._mobius_midpoint(v=torch.stack(masked_v)), a=torch.stack(masked_a))
        return torch.stack(h)

    def forward(self, input_vectors, mask):
        # project the hyperbolic vector to poincare model.
        poincare_in = PoincareBall().proj(x=input_vectors, c=self.curvature)
        att_coeff = self.input_linear(poincare_in)
        att_vectors = self.graph_attention(a=att_coeff, v=poincare_in, mask=self.mask)
        return PoincareBall().to_hyperboloid(x=self.att_out_linear(att_vectors), c=self.curvature)