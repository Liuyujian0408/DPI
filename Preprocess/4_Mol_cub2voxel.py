import glob
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_cub_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    total_points = lines[1].split(' ')[7]
    # Skip the first two title lines
    lines = lines[2:]

    # Read the atom count and origin
    atom_count, origin = int(lines[0].split()[0]), np.array([float(x) for x in lines[0].split()[1:4]])

    # Read the grid size and spacing
    grid_info = []
    for i in range(1, 4):
        parts = lines[i].split()
        grid_info.append((int(parts[0]), np.array([float(x) for x in parts[1:4]])))

    # Read the atom positions (we skip these for now)
    atom_positions = []
    for i in range(4, 4 + atom_count):
        parts = lines[i].split()
        atom_positions.append((int(parts[0]), np.array([float(x) for x in parts[1:4]])))

    # Read the density data
    density_data = []
    for line in lines[4 + atom_count:]:
        density_data.extend([float(x) for x in line.split()])

    assert len(density_data) == int(total_points)

    # Convert density data to NumPy array
    density_data = np.array(density_data)

    # Reshape density data based on grid dimensions
    nx, ny, nz = grid_info[0][0], grid_info[1][0], grid_info[2][0]
    # x = np.arange(density_data.size)
    #
    # plt.scatter(x, density_data)
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.title('2D Scatter Plot of 1D ndarray')
    # plt.show()



    density_data = density_data.reshape((nx, ny, nz))

    return density_data, origin, grid_info, atom_positions


def generate_coordinates(nx, ny, nz, origin, grid_info):
    dx, dy, dz = grid_info[0][1], grid_info[1][1], grid_info[2][1]

    x = origin[0] + np.arange(nx) * dx[0]
    y = origin[1] + np.arange(ny) * dy[1]
    z = origin[2] + np.arange(nz) * dz[2]

    return x, y, z
def convert_point_clound(density_data):
    x, y, z = np.nonzero(density_data)


    point_cloud = np.column_stack((x, y, z))


    print(f"Point cloud shape: {point_cloud.shape}")
    print(f"Sample points:\n{point_cloud[:5]}")
    df = pd.DataFrame(point_cloud, columns=['X', 'Y', 'Z'])
    df.to_csv('point_cloud.csv', index=False)
    print("Point cloud saved to 'point_cloud.csv'")


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.title('Point Cloud from Voxel Data')
    # plt.show()


def sparse_desity_and_norm_by_17915(voxel_data, origin, save_path, fixed_origin=(0, 0, 0)):


    # setting fixed number to select point
    all_indices = np.argwhere(voxel_data > -np.inf)
    all_values = voxel_data[all_indices[:, 0], all_indices[:, 1], all_indices[:, 2]]

    num_points = 17915

    if all_values.shape[0] < num_points:
        raise ValueError("not enough 17915")
    sorted_indices = np.argsort(all_values)[-num_points:]
    selected_indices = all_indices[sorted_indices]


    values = voxel_data[selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]]
    points = selected_indices + np.array(fixed_origin)
    point_info = {'values': values, 'data': points}
    with open(save_path, 'wb') as f:
        pickle.dump(point_info, f)


# Example usage
density_fold = 'molden'

output_point_fold = 'point_cloud_17915'
# os.makedirs(output_voxel_fold, exist_ok=True)
os.makedirs(output_point_fold, exist_ok=True)
data_paths = glob.glob(os.path.join(density_fold, '*/*/*/density.cub'))
for data_path in data_paths:
    key = data_path.split('/')[-2].split('_')[0]
    density_path = data_path
    density_data, origin, grid_info, atom_positions = read_cub_file(density_path)
    sparse_desity_and_norm_by_17915(density_data, origin, os.path.join(output_point_fold, key))

    # convert_point_clound(density_data)