import os
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import time
from tqdm import tqdm

# 范德华半径字典，单位是皮米(pm)
VAN_DER_WAALS_RADII_PM = {
    'H': 120, 'C': 170, 'N': 155, 'O': 152, 'S': 180, 'HE': 140, 'BE': 153, 'B': 192, 'GA': 187,
    'P': 180, 'F': 147, 'CL': 175, 'BR': 185, 'I': 198, 'ZN': 210, 'MG': 173, 'LI': 182, 'SR': 249,
    'K': 275, 'CA': 231, 'CU': 140, 'NI': 163, 'FE': 200, 'MN': 200, 'TI': 200, 'CR': 200, 'SI': 210,
    'AL': 184, 'NA': 227, 'NE': 154, 'AR': 188, 'SE': 190, 'CO': 200, 'HG': 155, 'CD': 158, 'CS': 343
    # 添加更多元素
}

# 将范德华半径从皮米转换为埃（Å）
VAN_DER_WAALS_RADII_A = {k: v / 100 for k, v in VAN_DER_WAALS_RADII_PM.items()}


def read_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    atom_count = int(lines[0].split()[0])
    atoms = []

    for line in lines[2:2 + atom_count]:
        parts = line.split()
        symbol = parts[0]
        x, y, z = map(float, parts[1:])
        atoms.append((symbol, x, y, z))

    return atoms


def sample_atom_volume(center, N, var, radius):
    samples = []
    while len(samples) < N:
        cov = [[var, 0, 0], [0, var, 0], [0, 0, var]]
        x = np.random.multivariate_normal(center, cov, size=N)
        valid_samples = [s for s in x if np.linalg.norm(s - np.array(center)) <= radius]
        samples.extend(valid_samples)
    return np.array(samples[:N])


def sample_within_vdw_radius(atom, center, num_samples):
    radius = VAN_DER_WAALS_RADII_A[atom]
    var = (radius / 3) ** 2
    return sample_atom_volume(center, num_samples, var, radius)


def sample_between_vdw_radii(atom, center, num_samples):
    radius_inner = VAN_DER_WAALS_RADII_A[atom]
    radius_outer = 2 * radius_inner
    var = ((radius_outer - radius_inner) / 3) ** 2
    samples = []
    while len(samples) < num_samples:
        cov = [[var, 0, 0], [0, var, 0], [0, 0, var]]
        x = np.random.multivariate_normal(center, cov, size=num_samples)
        valid_samples = [s for s in x if radius_inner < np.linalg.norm(s - np.array(center)) <= radius_outer]
        samples.extend(valid_samples)
    return np.array(samples[:num_samples])


def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


if __name__ == "__main__":

    save_path = 'DPI_pocket_cloud_fixed(cluster30)_num_multivariate_normal'
    total_samples_per_atom = 30

    path = 'DPI_xyz/DPI_pockets_xyz'
    dirs_list = os.listdir(path)
    strat_time = time.time()
    for dir in dirs_list:
        dirs = os.path.join(path, dir)
        subdir_list = os.listdir(os.path.join(path, dir))
        pock_list = []
        if dir == 'Test':
            for subsubdirs in subdir_list:
                pock_dirs = os.listdir(os.path.join(dirs, subsubdirs))
                for pock_id in pock_dirs:
                    pock_list.append(os.path.join(dirs, subsubdirs, pock_id))
        else:
            pock_dirs = os.listdir(dirs)
            for pock_id in pock_dirs:
                pock_list.append(os.path.join(dirs, pock_id))
        for id in tqdm(pock_list,desc=f'processing {dir}...'):
            if dir == 'Test':
                last_path = '\\'.join(id.split('\\')[1:-1])
            else:
                last_path = os.path.dirname(id).split('\\', -1)[-1]
            pocket_name = id.split('\\')[-1]
            output_dir = os.path.join(save_path, last_path)
            os.makedirs(output_dir, exist_ok=True)
            output_file_path = os.path.join(output_dir, f'{pocket_name}.pkl')

            xyz_path = os.path.join(id, f'{pocket_name}_pocket.xyz')
            atoms = read_xyz(xyz_path)

            samples_dict = {}
            all_points = []
            idx_list = []
            atom_centers = []
            distances = []

            atom_symbols = set([atom[0] for atom in atoms])
            exist_VAN_DER_WAALS_RADII_A = {k: VAN_DER_WAALS_RADII_A[k] for k in VAN_DER_WAALS_RADII_A if k in atom_symbols}
            for i, (symbol, x, y, z) in enumerate(atoms):
                atom_coord = [x, y, z]

                if symbol in VAN_DER_WAALS_RADII_A:

                    vdw_radius = VAN_DER_WAALS_RADII_A[symbol]
                    num_samples_within = int(total_samples_per_atom * (vdw_radius / max(exist_VAN_DER_WAALS_RADII_A.values())))
                    num_samples_between = total_samples_per_atom - num_samples_within
                    if num_samples_between <= 0:
                        num_samples_within -= 1
                        num_samples_between += 1

                    samples_within = sample_within_vdw_radius(symbol, atom_coord, num_samples_within)
                    samples_between = sample_between_vdw_radii(symbol, atom_coord, num_samples_between)

                    all_samples = np.vstack((samples_within, samples_between))

                    for sample in all_samples:
                        distance = calculate_distance(sample, atom_coord)
                        samples_dict[tuple(sample)] = {
                            "point xyz_data": list(sample),
                            "atom": i,
                            "distance": distance
                        }
                        all_points.append(sample)
                        idx_list.append(i)
                        atom_centers.append(atom_coord)
                        distances.append(distance)

            all_points = np.array(all_points)
            idx_list = np.array(idx_list)
            atom_centers = np.array(atom_centers)
            distances = np.array(distances)


            # 保存数据到文件
            data_to_save = {
                'points': all_points,
                'indices': idx_list,
                'atom_centers': atom_centers,
                'distances': distances
            }
            with open(output_file_path, 'wb') as f:
                pickle.dump(data_to_save, f)

            # 可视化3D点云
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            unique_indices = np.unique(idx_list)
            colors = {i: (random.random(), random.random(), random.random()) for i in unique_indices}
            sample_colors = np.array([colors[i] for i in idx_list])

            ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], s=1, c=sample_colors, alpha=0.4)

            centers = np.array([atom[1:] for atom in atoms])
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=50, c='black', marker='x', alpha=1)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()


            #print(f'Saved data for {pocket_id} with {len(all_points)} points.')
    end_time = time.time()
    waste_time = end_time - strat_time
    print(f'Elapsed time {waste_time:.4f}s')