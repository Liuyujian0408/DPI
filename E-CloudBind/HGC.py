# copy from the source code of dgl and make some modification.

"""Heterograph NN modules"""
from functools import partial
import torch as th
import torch.nn as nn


__all__ = ['HeteroGraphConv', 'HeteroLinear', 'HeteroEmbedding']

class HeteroGraphConv(nn.Module):
    r"""
    >>> import dgl
    >>> g = dgl.heterograph({
    ...     ('user', 'follows', 'user') : edges1,
    ...     ('user', 'plays', 'game') : edges2,
    ...     ('store', 'sells', 'game')  : edges3})

    >>> import dgl.nn.pytorch as dglnn
    >>> conv = dglnn.HeteroGraphConv({
    ...     'follows' : dglnn.GraphConv(...),
    ...     'plays' : dglnn.GraphConv(...),
    ...     'sells' : dglnn.SAGEConv(...)},
    ...     aggregate='sum')

    >>> import torch as th
    >>> h1 = {'user' : th.randn((g.number_of_nodes('user'), 5))}
    >>> h2 = conv(g, h1)
    >>> f1 = {'user' : ..., 'store' : ...}
    >>> f2 = conv(g, f1)
    >>> print(f2.keys())
    >>> g1 = {'store' : ...}
    >>> g2 = conv(g, g1)
    >>> print(g2.keys())
    >>> x_src = {'user' : ..., 'store' : ...}
    >>> x_dst = {'user' : ..., 'game' : ...}
    >>> y_dst = conv(g, (x_src, x_dst))
    >>> print(y_dst.keys())
    """
    def __init__(self, mods, aggregate='sum'):
        super(HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)
        # Do not break if graph has 0-in-degree nodes.
        # Because there is no general rule to add self-loop for heterograph.
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
        if isinstance(aggregate, str):
            self.agg_fn = get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregate

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):

        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty : [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph,
                    (src_inputs[stype], dst_inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                
                edge_feat = rel_graph.edata['e']
                if stype not in inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph,
                    (inputs[stype], inputs[dtype]),
                    edge_feat,
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts


def _max_reduce_func(inputs, dim):
    return th.max(inputs, dim=dim)[0]

def _min_reduce_func(inputs, dim):
    return th.min(inputs, dim=dim)[0]

def _sum_reduce_func(inputs, dim):
    return th.sum(inputs, dim=dim)

def _mean_reduce_func(inputs, dim):
    return th.mean(inputs, dim=dim)

def _stack_agg_func(inputs, dsttype):
    if len(inputs) == 0:
        return None
    return th.stack(inputs, dim=1)

def _agg_func(inputs, dsttype, fn):
    if len(inputs) == 0:
        return None
    stacked = th.stack(inputs, dim=0)
    return fn(stacked, dim=0)

def get_aggregate_fn(agg):
    if agg == 'sum':
        fn = _sum_reduce_func
    elif agg == 'max':
        fn = _max_reduce_func
    elif agg == 'min':
        fn = _min_reduce_func
    elif agg == 'mean':
        fn = _mean_reduce_func
    elif agg == 'stack':
        fn = None  # will not be called
    else:
        print('Invalid cross type aggregator. Must be one of '
                       '"sum", "max", "min", "mean" or "stack". But got "%s"' % agg)
    if agg == 'stack':
        return _stack_agg_func
    else:
        return partial(_agg_func, fn=fn)

class HeteroLinear(nn.Module):
    """
    >>> import dgl
    >>> import torch
    >>> from dgl.nn import HeteroLinear

    >>> layer = HeteroLinear({'user': 1, ('user', 'follows', 'user'): 2}, 3)
    >>> in_feats = {'user': torch.randn(2, 1), ('user', 'follows', 'user'): torch.randn(3, 2)}
    >>> out_feats = layer(in_feats)
    >>> print(out_feats['user'].shape)
    >>> print(out_feats[('user', 'follows', 'user')].shape)
    """
    def __init__(self, in_size, out_size, bias=True):
        super(HeteroLinear, self).__init__()

        self.linears = nn.ModuleDict()
        for typ, typ_in_size in in_size.items():
            self.linears[str(typ)] = nn.Linear(typ_in_size, out_size, bias=bias)

    def forward(self, feat):

        out_feat = dict()
        for typ, typ_feat in feat.items():
            out_feat[typ] = self.linears[str(typ)](typ_feat)

        return out_feat


class HeteroEmbedding(nn.Module):
    """

    >>> import dgl
    >>> import torch
    >>> from dgl.nn import HeteroEmbedding

    >>> layer = HeteroEmbedding({'user': 2, ('user', 'follows', 'user'): 3}, 4)
    >>> # Get the heterogeneous embedding table
    >>> embeds = layer.weight
    >>> print(embeds['user'].shape)
    >>> print(embeds[('user', 'follows', 'user')].shape)

    >>> # Get the embeddings for a subset
    >>> input_ids = {'user': torch.LongTensor([0]),
    ...              ('user', 'follows', 'user'): torch.LongTensor([0, 2])}
    >>> embeds = layer(input_ids)
    >>> print(embeds['user'].shape)
    >>> print(embeds[('user', 'follows', 'user')].shape)
    """
    def __init__(self, num_embeddings, embedding_dim):
        super(HeteroEmbedding, self).__init__()

        self.embeds = nn.ModuleDict()
        self.raw_keys = dict()
        for typ, typ_num_rows in num_embeddings.items():
            self.embeds[str(typ)] = nn.Embedding(typ_num_rows, embedding_dim)
            self.raw_keys[str(typ)] = typ

    @property
    def weight(self):

        return {self.raw_keys[typ]: emb.weight for typ, emb in self.embeds.items()}

    def forward(self, input_ids):

        embeds = dict()
        for typ, typ_ids in input_ids.items():
            embeds[typ] = self.embeds[str(typ)](typ_ids)

        return embeds