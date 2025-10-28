import os
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from rdkit import Chem
from sklearn.cluster import KMeans
import multiprocessing
from itertools import repeat
def read_point_clouds_and_atoms(point_cloud_dir, mol_dir):
    point_clouds_dict = {}
    num_atoms_dict = {}

    for filename in os.listdir(point_cloud_dir):
        try:
            filepath_point = os.path.join(point_cloud_dir, filename)
            filepath_mol = os.path.join(mol_dir, filename, f"{filename}_ligand.mol2")
            with open(filepath_point, 'rb') as f:
                data = pickle.load(f)
                points = data['data']
                values = data['values']

            mol = Chem.MolFromMol2File(filepath_mol)
            num_atoms = mol.GetNumAtoms()

            num_atoms_dict[filename] = num_atoms
            point_clouds_dict[filename] = (points, values)
        except:
            print(f"{filepath_mol} is not exist")
    return point_clouds_dict, num_atoms_dict


def balance_clusters_with_overlap(points, values, num_clusters, target_cluster_size):

    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(points)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    balanced_clusters = []
    balanced_values = []
    cluster_indices = []
    cluster_centers_list = []

    for i in range(num_clusters):
        cluster_points = points[labels == i]
        cluster_values = values[labels == i]

        if len(cluster_points) < target_cluster_size:
            num_to_add = target_cluster_size - len(cluster_points)
            nbrs = NearestNeighbors(n_neighbors=num_to_add).fit(points)
            distances, indices = nbrs.kneighbors(cluster_centers[i].reshape(1, -1))
            additional_points = points[indices[0][:num_to_add]]
            additional_values = values[indices[0][:num_to_add]]
            cluster_points = np.concatenate((cluster_points, additional_points))
            cluster_values = np.concatenate((cluster_values, additional_values))
        elif len(cluster_points) > target_cluster_size:
            num_to_remove = len(cluster_points) - target_cluster_size
            distances = np.linalg.norm(cluster_points - cluster_centers[i], axis=1)
            farthest_indices = np.argsort(distances)[-num_to_remove:]
            cluster_points = np.delete(cluster_points, farthest_indices, axis=0)
            cluster_values = np.delete(cluster_values, farthest_indices, axis=0)

        balanced_clusters.append(cluster_points)
        balanced_values.append(cluster_values)
        cluster_indices.extend([i] * len(cluster_points))

        distances_to_center = np.linalg.norm(cluster_points - cluster_centers[i], axis=1)
        closest_point_idx = np.argmin(distances_to_center)
        center_point = cluster_points[closest_point_idx]
        cluster_centers_list.append(center_point)
    return (np.vstack(balanced_clusters),
            np.vstack(balanced_values).flatten(),
            np.array(cluster_indices),
            np.array(cluster_centers_list))

def balance(point_clouds, num_atoms, final_cluster_size, save_dir):

    name = point_clouds[0]
    points = point_clouds[1]
    values = point_clouds[2]
    print(f'{name}')

    clusters, cluster_values, cluster_indices, \
        cluster_centers = balance_clusters_with_overlap(points,
                                                        values,
                                                        num_atoms,
                                                        final_cluster_size)

    data_to_save = {
        'data': clusters,
        'value': cluster_values,
        'idx_cluster': cluster_indices,
        'centers': cluster_centers
    }

    save_path = os.path.join(save_dir, f"{name}")
    with open(save_path, 'wb') as f:
        pickle.dump(data_to_save, f)


def main():
    point_cloud_dir = "point_cloud_17915"
    mol_dir = "DPI_data"
    output_point_cloud_dir = "point_cloud_with_clus"
    os.makedirs(output_point_cloud_dir, exist_ok=True)

    final_cluster_size = 400
    print(f"Final cluster size: {final_cluster_size}")

    for id in os.listdir(point_cloud_dir):
        point_path = point_cloud_dir
        mol_path = mol_dir
        point_clouds_dict, num_atoms_dict = read_point_clouds_and_atoms(point_path, mol_path)
        point_clouds = [[name, points, values] for name, (points, values) in point_clouds_dict.items()]
        num_atoms = [num_atoms_dict[point_cloud[0]] for point_cloud in point_clouds if point_cloud[0] in num_atoms_dict]
        final_cluster = repeat(final_cluster_size, len(num_atoms))
        os.makedirs(os.path.join(output_point_cloud_dir), exist_ok=True)
        save_dir = repeat(os.path.join(output_point_cloud_dir), len(num_atoms))
        pool = multiprocessing.Pool(64)
        pool.starmap(balance, zip(point_clouds, num_atoms, final_cluster, save_dir))
        pool.close()
        pool.join()




if __name__ == "__main__":
    main()