# %%
import os
import numpy as np
import pickle
import torch
import dgl
from rdkit import RDLogger
import warnings
import tqdm

RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')



def point_constrctor(val_df, valid_dir):
    point_dir = valid_dir
    mol_point_paths = {}
    pocket_point_paths = {}
    pocket_xyz = {}


    for i, row in val_df.iterrows():
        cid, pKa = row['pdbid'], float(row['-logKd/Ki'])

        mol_point_paths[cid] = os.path.join(point_dir, cid,
                                            'point_cloud_with_clus',
                                            cid)
        pocket_point_paths[cid] = os.path.join(point_dir, cid,
                                               'DPI_pocket_cloud_fixed_cluster_num_multivariate_normal',
                                               f"{cid}.pkl")
        pocket_xyz[cid] = os.path.join(point_dir, cid,
                                       'DPI_complex_pkl',
                                       f"{cid}.pkl")

    val_paths = [mol_point_paths, pocket_point_paths, pocket_xyz]

    return val_paths



# %
def collate_fn(data_batch):
    """
    used for dataset generated
    :param data_batch:
    :return:
    """
    # print(data_batch)
    g, label, mol_point_xyz, clu_indices_mol, pocket_point_xyz, clu_indices_pock, pocket_mol_idx, keys = map(list,
                                                                                                       zip(*data_batch))
    bg = dgl.batch(g)
    y = torch.cat(label, dim=0)

    return bg, y, mol_point_xyz, clu_indices_mol, pocket_point_xyz, clu_indices_pock, pocket_mol_idx, keys


class GraphDataset(object):
    """
    This class is used for generating graph objects using multi process
    """

    def __init__(self, data_dir,
                 data_df,
                 dis_threshold=5.0,
                 graph_type='',
                 num_process=48,
                 point_path=None,
                 create=True):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.num_process = num_process
        self.protein_keys = None
        if point_path is not None:
            self.point_path = point_path
            self._pre_process_point(point_path)
            self.protein_keys = self.point_cache.keys()
            self._pre_process_complex()
        else:
            self._pre_process_complex()

    def _pre_process_complex(self):

        data_dir = self.data_dir
        data_df = self.data_df
        graph_type = self.graph_type

        graph_path_list = []
        if self.protein_keys is not None:
            for i, row in data_df.iterrows():
                cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
                if cid in self.protein_keys:
                    if isinstance(data_dir, list) is not True:
                        complex_dir = os.path.join(data_dir, cid)
                    else:
                        complex_dir = os.path.join(data_dir[0], cid)
                        if os.path.exists(complex_dir) is not True:
                            complex_dir = os.path.join(data_dir[1], cid)
                    graph_path = os.path.join(complex_dir, f"{graph_type}-{cid}.dgl")
                    graph_path_list.append((cid, graph_path))
        else:
            for i, row in data_df.iterrows():
                cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
                complex_dir = os.path.join(data_dir, cid)
                graph_path = os.path.join(complex_dir, f"{graph_type}-{cid}.dgl")

                graph_path_list.append((cid, graph_path))

        self.graph_paths = graph_path_list

    def _pre_process_point(self, point_path):
        mol_point_paths, pocket_point_paths, pocket_xyz_paths = point_path[0], point_path[1], point_path[2]
        self.point_cache = {}
        for idx in tqdm.tqdm(list(mol_point_paths.keys())):
            if os.path.exists(pocket_xyz_paths[idx]) and os.path.exists(mol_point_paths[idx]):
                with open(pocket_xyz_paths[idx], 'rb') as f:
                    pocket_ = pickle.load(f)[1]
                if pocket_ is not None:
                    with open(mol_point_paths[idx], 'rb') as f:
                        mol_point = pickle.load(f)
                    f.close()
                    with open(pocket_point_paths[idx], 'rb') as f:
                        pocket_point = pickle.load(f)
                    f.close()
                    with open(pocket_xyz_paths[idx], 'rb') as f:
                        pocket_xyz = pickle.load(f)[1].GetConformers()[0].GetPositions()
                    f.close()
                    clu_indices_pock = np.unique(pocket_point['atom_centers'], axis=0)
                    pocket_idx = self.find_indices(pocket_xyz, clu_indices_pock)
                    pocket_point['centers_idx'] = pocket_idx

                    mol_point.pop('value')
                    mol_point.pop('centers')
                    pocket_point.pop('distances')
                    pocket_point.pop('atom_centers')

                    self.point_cache[idx] = [mol_point, pocket_point]

        print(f'point clound data size is {len(self.point_cache.keys())}')

    def find_indices(self, A, B):

        diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]

        dist = np.sum(diff ** 2, axis=-1)

        indices = np.argmin(dist, axis=1)
        return indices

    def __getitem__(self, idx):
        keys, paths = self.graph_paths[idx][0], self.graph_paths[idx][1]
        complex, label = torch.load(paths)[0], torch.load(paths)[1]

        mol_point, poc_point = self.point_cache[keys][0], self.point_cache[keys][1]
        # mol point cloud
        mol_point_xyz = mol_point['data']
        clu_indices_mol = mol_point['idx_cluster']

        # pocket points cloud
        pocket_point_xyz = poc_point['points']
        clu_indices_pock = poc_point['indices']
        pocket_mol_idx = poc_point['centers_idx']

        return complex, label, mol_point_xyz, clu_indices_mol, pocket_point_xyz, clu_indices_pock, pocket_mol_idx, keys

    def __len__(self):
        return len(self.graph_paths)




