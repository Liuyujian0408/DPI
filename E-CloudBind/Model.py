import torch.nn as nn
from dgl.nn.pytorch import edge_softmax
import dgl
from CIGConv import CIGConv
from NIGConv import NIGConv
from HGC import HeteroGraphConv

from model_gcn3d import GCN3DV2
import torch
import torch.nn.functional as F



class CrossAttention(nn.Module):    # Actually is linear
    def __init__(self, input_dim1, input_dim2, hidden_dim, dropout_prob):
        super(CrossAttention, self).__init__()
        self.linear1 = nn.Linear(input_dim1, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, tensor1, tensor2):

        query = self.linear1(tensor1)  # (n1, 256)
        query = self.dropout(query)
        output = torch.cat((query, tensor2), dim=1)
        # output = query * key


        return output

class ECloudBind(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_feat_size, layer_num=3, droup_out=0.):
        super(ECloudBind, self).__init__()

        self.gcn3d = GCN3DV2(support_num=1, neighbor_num=5, pooling_rate=4)
        self.crossattention = CrossAttention(input_dim1=node_feat_size,
                                             input_dim2=128,
                                             hidden_dim=128,
                                             dropout_prob=droup_out)

        self.convs = nn.ModuleList()
        for _ in range(layer_num):
            convl = CIGConv(hidden_feat_size, hidden_feat_size, drop=droup_out)
            convp = CIGConv(hidden_feat_size, hidden_feat_size, drop=droup_out)
            convlp = NIGConv(hidden_feat_size, hidden_feat_size, feat_drop=droup_out)
            convpl = NIGConv(hidden_feat_size, hidden_feat_size, feat_drop=droup_out)
            conv = HeteroGraphConv(
                {
                    'intra_l': convl,
                    'intra_p': convp,
                    'inter_l2p': convlp,
                    'inter_p2l': convpl
                }
            )
            self.convs.append(conv)

        self.lin_edge_ll = nn.Linear(edge_feat_size, hidden_feat_size)
        self.lin_edge_pp = nn.Linear(edge_feat_size, hidden_feat_size)

        self.lin_edge_lp = nn.Linear(11, hidden_feat_size)
        self.lin_edge_pl = nn.Linear(11, hidden_feat_size)

        # atom-atom affinities
        self.inter_atompairs = AtomAtomAffinities(hidden_feat_size, hidden_feat_size, hidden_feat_size)

        # bias correction
        self.bias_ligandpocket = BiasCorrectionLigandPocket(hidden_feat_size, hidden_feat_size, hidden_feat_size)
        self.bias_pocketligand = BiasCorrectionPocketLigand(hidden_feat_size, hidden_feat_size, hidden_feat_size)

    def rearrange_points(self, A, B, S):
        all_clusters = []
        batch_indices = []

        for batch_idx, (points, clusters) in enumerate(zip(A, B)):
            unique_clusters = clusters.unique()
            for cluster in unique_clusters:
                cluster_points = points[clusters == cluster]
                original_indices = torch.where(clusters == cluster)[0]

                if cluster_points.shape[0] == S:
                    all_clusters.append(cluster_points)

                batch_indices.extend([batch_idx] * S)

        return torch.stack(all_clusters), batch_indices

    def restore_original_tensor(self, clusters, batch_indices, original_shapes, S):

        restored_tensors = [None] * len(original_shapes)
        count = [0] * len(original_shapes)

        batch_clusters = [[] for _ in range(len(original_shapes))]

        for i, cluster in enumerate(clusters):
            batch_idx = batch_indices[i * S]
            batch_clusters[batch_idx].append(cluster)

        for batch_idx, cluster_list in enumerate(batch_clusters):
            if cluster_list:
                restored_tensors[batch_idx] = torch.stack(cluster_list)

        return restored_tensors

    def forward_point(self, x, indices, device):
        mol_xyz, pocket_xyz = x[0], x[1]
        mol_indices, poc_indices, pocket_mol_idx = indices[0], indices[1], indices[2]

        mol_original_shapes = [a.shape[0] for a in mol_xyz]
        pock_original_shapes = [a.shape[0] for a in pocket_xyz]
        original_shapes = [mol_original_shapes, pock_original_shapes]
        mol_input, mol_batch_indices = self.rearrange_points(mol_xyz,
                                                             mol_indices,
                                                             S=400)
        poc_input, poc_batch_indices = self.rearrange_points(pocket_xyz,
                                                             poc_indices,
                                                             S=30)

        mol_feature = self.gcn3d(mol_input)
        poc_feature = self.gcn3d(poc_input)

        mol_restored_tensors = self.restore_original_tensor(mol_feature,
                                                            mol_batch_indices,
                                                            original_shapes[0],
                                                            S=400)
        poc_restored_tensors = self.restore_original_tensor(poc_feature,
                                                            poc_batch_indices,
                                                            original_shapes[1],
                                                            S=30)


        return mol_restored_tensors, poc_restored_tensors, pocket_mol_idx

    def forward(self, bg, xyz, indices, device):
        mol_num, poc_num = bg.ndata['h']['ligand'].shape[0], bg.ndata['h']['pocket'].shape[0]

        mol_restored_tensors, poc_restored_tensors, pocket_mol_idx = self.forward_point(xyz, indices, device)

        poc_features = torch.cat([torch.index_select(feature, 0, mol_idx) for feature, mol_idx in
                        zip(poc_restored_tensors, pocket_mol_idx)])

        poc_features = F.interpolate(poc_features.unsqueeze(0).reshape(1, 128, -1), size=poc_num, mode='linear',
                               align_corners=False).reshape(1, -1, 128).squeeze(0)

        mol_features = torch.cat(mol_restored_tensors, dim=0)

        if mol_features.shape[0] != mol_num:
            mol_features = F.interpolate(mol_features.unsqueeze(0).reshape(1, 128, -1),
                                          size=mol_num, mode='linear', align_corners=False).reshape(1, -1, 128).squeeze(0)

        point_feature = torch.cat((mol_features, poc_features), dim=0)


        atom_feats = torch.cat((bg.ndata['h']['ligand'], bg.ndata['h']['pocket']), dim=0)
        new_atom_feats = self.crossattention(atom_feats, point_feature)
        # atom_feats = bg.ndata['h']
        bond_feats = bg.edata['e']
        atom_feats = {
            'ligand': new_atom_feats[:mol_num, :],
            'pocket': new_atom_feats[mol_num:, :]
        }

        bond_feats = {
            ('ligand', 'intra_l', 'ligand'): self.lin_edge_ll(bond_feats[('ligand', 'intra_l', 'ligand')]),
            ('pocket', 'intra_p', 'pocket'): self.lin_edge_pp(bond_feats[('pocket', 'intra_p', 'pocket')]),
            ('ligand', 'inter_l2p', 'pocket'): self.lin_edge_lp(bond_feats[('ligand', 'inter_l2p', 'pocket')]),
            ('pocket', 'inter_p2l', 'ligand'): self.lin_edge_pl(bond_feats[('pocket', 'inter_p2l', 'ligand')]),
        }

        bg.edata['e'] = bond_feats

        rsts = atom_feats
        for conv in self.convs:
            rsts = conv(bg, rsts)

        bg.nodes['ligand'].data['h'] = rsts['ligand']
        bg.nodes['pocket'].data['h'] = rsts['pocket']

        # atom-atom affinities
        atompairs_lp, atompairs_pl, total_scores = self.inter_atompairs(bg)

        # bias correction
        bias_lp = self.bias_ligandpocket(bg)
        bias_pl = self.bias_pocketligand(bg)

        return (atompairs_lp - bias_lp).view(-1), (atompairs_pl - bias_pl).view(-1), total_scores


class AtomAtomAffinities(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_feat_size):
        super(AtomAtomAffinities, self).__init__()
        self.prj_lp_src = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_lp_dst = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_lp_edge = nn.Linear(edge_feat_size, hidden_feat_size)

        self.prj_pl_src = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_pl_dst = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_pl_edge = nn.Linear(edge_feat_size, hidden_feat_size)

        self.fc_lp = nn.Linear(hidden_feat_size, 1)
        self.fc_pl = nn.Linear(hidden_feat_size, 1)

    def apply_interactions(self, edges):
        return {'i': edges.data['e'] * edges.src['h'] * edges.dst['h']}

    def forward(self, g):
        with g.local_scope():
            node_ligand_feats = g.nodes['ligand'].data['h']
            node_pocket_feats = g.nodes['pocket'].data['h']
            edge_lp_feat = g.edges['inter_l2p'].data['e']
            edge_pl_feat = g.edges['inter_p2l'].data['e']

            g.nodes['ligand'].data['h'] = self.prj_lp_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.prj_lp_dst(node_pocket_feats)
            g.edges['inter_l2p'].data['e'] = self.prj_lp_edge(edge_lp_feat)

            g.apply_edges(self.apply_interactions, etype='inter_l2p')
            logit_lp_ = self.fc_lp(g.edges['inter_l2p'].data['i'])
            g.edges['inter_l2p'].data['logit_lp'] = logit_lp_
            logit_lp = dgl.sum_edges(g, 'logit_lp', etype='inter_l2p')

            etype = ('ligand', 'inter_l2p', 'pocket')
            src_nodes = g.edges(etype=etype)[0]
            lp_scores = logit_lp_.squeeze(1)

            total_scores_lp = torch.zeros(g.num_nodes('ligand'), device=lp_scores.device)

            total_scores_lp.scatter_add_(0, src_nodes, lp_scores)

            g.nodes['ligand'].data['h'] = self.prj_pl_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.prj_pl_dst(node_pocket_feats)
            g.edges['inter_p2l'].data['e'] = self.prj_pl_edge(edge_pl_feat)
            g.apply_edges(self.apply_interactions, etype='inter_p2l')
            logit_pl_ = self.fc_pl(g.edges['inter_p2l'].data['i'])
            g.edges['inter_p2l'].data['logit_pl'] = logit_pl_
            logit_pl = dgl.sum_edges(g, 'logit_pl', etype='inter_p2l')

            etype = ('ligand', 'inter_l2p', 'pocket')
            src_nodes = g.edges(etype=etype)[0]
            pl_scores = logit_pl_.squeeze(1)

            total_scores_pl = torch.zeros(g.num_nodes('ligand'), device=pl_scores.device)

            total_scores_pl.scatter_add_(0, src_nodes, pl_scores)

            total_scores = total_scores_pl + total_scores_lp

            return logit_lp, logit_pl, total_scores


class BiasCorrectionLigandPocket(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_feat_size):
        super(BiasCorrectionLigandPocket, self).__init__()
        self.prj_src = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_dst = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_edge = nn.Linear(edge_feat_size, hidden_feat_size)

        self.w_src = nn.Linear(node_feat_size, hidden_feat_size)
        self.w_dst = nn.Linear(node_feat_size, hidden_feat_size)
        self.w_edge = nn.Linear(edge_feat_size, hidden_feat_size)

        self.lin_att = nn.Sequential(
            nn.PReLU(),
            nn.Linear(hidden_feat_size, 1)
        )

        self.fc = FC(hidden_feat_size, 200, 2, 0.1, 1)

    def get_weight(self, edges):
        w = edges.src['h'] + edges.dst['h'] + edges.data['e']
        w = self.lin_att(w)

        return {'w': w}

    def apply_scores(self, edges):
        return {'l': edges.data['a'] * edges.data['e'] * edges.src['h'] * edges.dst['h']}

    def forward(self, g):
        with g.local_scope():
            node_ligand_feats = g.nodes['ligand'].data['h']
            node_pocket_feats = g.nodes['pocket'].data['h']
            edge_feat = g.edges['inter_l2p'].data['e']

            g.nodes['ligand'].data['h'] = self.prj_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.prj_dst(node_pocket_feats)
            g.edges['inter_l2p'].data['e'] = self.prj_edge(edge_feat)
            g.apply_edges(self.get_weight, etype='inter_l2p')
            scores = edge_softmax(g['inter_l2p'], g.edges['inter_l2p'].data['w'])
            g.edges['inter_l2p'].data['a'] = scores
            g.nodes['ligand'].data['h'] = self.w_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.w_dst(node_pocket_feats)
            g.edges['inter_l2p'].data['e'] = self.w_edge(edge_feat)
            g.apply_edges(self.apply_scores, etype='inter_l2p')

            hidden_feat = dgl.sum_edges(g, 'l', etype='inter_l2p')
            bias = self.fc(hidden_feat)

            return bias


class BiasCorrectionPocketLigand(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_feat_size):
        super(BiasCorrectionPocketLigand, self).__init__()
        self.prj_src = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_dst = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_edge = nn.Linear(edge_feat_size, hidden_feat_size)

        self.w_src = nn.Linear(node_feat_size, hidden_feat_size)
        self.w_dst = nn.Linear(node_feat_size, hidden_feat_size)
        self.w_edge = nn.Linear(edge_feat_size, hidden_feat_size)

        self.lin_att = nn.Sequential(
            nn.PReLU(),
            nn.Linear(hidden_feat_size, 1)
        )

        self.fc = FC(hidden_feat_size, 200, 2, 0.1, 1)

    def get_weight(self, edges):
        w = edges.src['h'] + edges.dst['h'] + edges.data['e']
        w = self.lin_att(w)

        return {'w': w}

    def apply_scores(self, edges):
        return {'l': edges.data['a'] * edges.data['e'] * edges.src['h'] * edges.dst['h']}

    def forward(self, g):
        with g.local_scope():
            node_ligand_feats = g.nodes['ligand'].data['h']
            node_pocket_feats = g.nodes['pocket'].data['h']
            edge_feat = g.edges['inter_p2l'].data['e']

            g.nodes['ligand'].data['h'] = self.prj_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.prj_dst(node_pocket_feats)
            g.edges['inter_p2l'].data['e'] = self.prj_edge(edge_feat)
            g.apply_edges(self.get_weight, etype='inter_p2l')
            scores = edge_softmax(g['inter_p2l'], g.edges['inter_p2l'].data['w'])

            g.edges['inter_p2l'].data['a'] = scores
            g.nodes['ligand'].data['h'] = self.w_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.w_dst(node_pocket_feats)
            g.edges['inter_p2l'].data['e'] = self.w_edge(edge_feat)
            g.apply_edges(self.apply_scores, etype='inter_p2l')

            bias = self.fc(dgl.sum_edges(g, 'l', etype='inter_p2l'))

            return bias


class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)

        return h
