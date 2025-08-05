
import os

from torch.utils.data import DataLoader
import pandas as pd

from graph_constructor_v2 import point_constrctor, GraphDataset, collate_fn
from Model import DTIPredictor
from utils import *

from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

import warnings
warnings.filterwarnings('ignore')
seed_everything(42)


def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for bg, label, mol_point_xyzs, clu_indices_mol, pocket_point_xyzs, clu_indices_pock, pocket_mol_idx in dataloader:
        mol_xyz = [torch.tensor(np.array(mol_point_xyz), dtype=torch.float32).to(device) for mol_point_xyz
                   in mol_point_xyzs]

        pocket_xyz = [torch.tensor(np.array(pocket_point_xyz), dtype=torch.float32).to(device) for
                      pocket_point_xyz in pocket_point_xyzs]

        mol_indices = [torch.tensor(indice, dtype=torch.int).to(device) for indice in clu_indices_mol]

        poc_indices = [torch.tensor(indice, dtype=torch.int).to(device) for indice in clu_indices_pock]
        pocket_mol_idx = [torch.as_tensor(indice, dtype=torch.int).to(device) for indice in pocket_mol_idx]

        bg, label = bg.to(device), label.to(device)
        xyz = (mol_xyz, pocket_xyz)
        indices = (mol_indices, poc_indices, pocket_mol_idx)
        with torch.no_grad():
            pred_lp, pred_pl = model(bg, xyz, indices, device)
            pred = (pred_lp + pred_pl) / 2
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    pr = pearsonr(pred, label)[0]
    rmse = np.sqrt(mean_squared_error(label, pred))

    model.train()

    return rmse, pr

if __name__ == '__main__':
    data_root = './data'

    toy_dir = os.path.join(data_root, 'toy')
    toy_df = pd.read_csv(os.path.join(data_root, 'toy.csv'))

    toy_paths = point_constrctor(toy_df)
    device = torch.device('cuda:0')

    model = DTIPredictor(node_feat_size=35, edge_feat_size=17,
                         hidden_feat_size=256, layer_num=1, droup_out=0.1).to(device)

    load_model_dict(model,
                    './weights/best_model_ECloudbind.pt')

    toy_set = GraphDataset(toy_dir, toy_df,
                             point_path=toy_paths, create=False)

    toy_loader = DataLoader(toy_set, batch_size=48, shuffle=False, collate_fn=collate_fn, num_workers=8)
    toy_rmse, toy_pr = val(model, toy_loader, device)

    msg = "valid_rmse-%.4f, valid_pr-%.4f " \
                % (toy_rmse, toy_pr)
    print(msg)
