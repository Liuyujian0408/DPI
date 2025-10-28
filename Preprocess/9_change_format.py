import os
import shutil
import pandas as pd
import pickle


data_df = pd.read_csv('toy_examples.csv')

for i, row in data_df.iterrows():
    cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
    save_dir = os.path.join('toy', cid)
    os.makedirs(save_dir, exist_ok=True)

    # dgl
    start_file = os.path.join('DPI_data', cid, f'Graph_E_cloudBind-{cid}.dgl')
    shutil.copyfile(start_file, os.path.join(save_dir, f'Graph_E_cloudBind-{cid}.dgl'))

    # DPI_pocket_cloud_fixed_cluster_num_multivariate_normal
    start_file = os.path.join('DPI_pocket_cloud_fixed_cluster_num_multivariate_normal', cid, f'{cid}_pocket.pkl')
    os.makedirs(os.path.join(save_dir, 'DPI_pocket_cloud_fixed_cluster_num_multivariate_normal'), exist_ok=True)
    shutil.copyfile(start_file, os.path.join(save_dir, 'DPI_pocket_cloud_fixed_cluster_num_multivariate_normal', f'{cid}.pkl'))

    # DPI_complex_pkl
    start_file = os.path.join('DPI_data', cid, f'{cid}.rdkit')
    dst_file = os.path.join(save_dir, 'DPI_complex_pkl', f'{cid}.pkl')
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    with open(start_file, 'rb') as f:
        mols = pickle.load(f)
    with open(dst_file, 'wb') as f:
        pickle.dump(mols, f)

    # point_cloud_with_clus
    start_file = os.path.join('point_cloud_with_clus', cid)
    os.makedirs(os.path.join(save_dir, 'point_cloud_with_clus'), exist_ok=True)
    shutil.copyfile(start_file, os.path.join(save_dir, 'point_cloud_with_clus', cid))

    # cp toy csv
    shutil.copyfile('toy_examples.csv', os.path.join(os.path.dirname(save_dir), 'toy_examples.csv'))




