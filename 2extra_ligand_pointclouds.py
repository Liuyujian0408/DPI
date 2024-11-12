import multiprocessing
import os
import shutil
import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing

def screen_density_cub(data_fold, out_fold):
    dirs_list = os.listdir(data_fold)
    for dir in dirs_list:
        dirs = os.path.join(data_fold, dir)
        subdir_list = os.listdir(dirs)
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
        exist_density = []
        for id in pock_list:
            print(f'processing {id}')
            if dir == 'Test':
                last_path = '/'.join(id.split('/')[2:-1])
            else:
                last_path = os.path.dirname(id).split('/', -1)[-1]

            pocket_name = id.split('/')[-1]
            cal_files = os.listdir(id)
            exist = any(file.endswith('.cub') for file in cal_files)
            if exist:
                exist_density.append(id)
                src_path = os.path.join(id, "density.cub")
                dst_path = os.path.join(output_fold, last_path, pocket_name, "density.cub")
                os.makedirs(os.path.join(output_fold, last_path, pocket_name), exist_ok=True)
                shutil.copy(src_path, dst_path)
            else:
                print(f'{pocket_name} is not density.cub, which need to delet')
        print(len(exist_density))



def read_cub_file(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        total_points = lines[1].split(' ')[7]
    else:
        print(f'{filename}does not exist')
        pass
    # Read the atom count and origin
    try:
        atom_count, origin = int(lines[2].split()[0]), np.array([float(x) for x in lines[2].split()[1:4]])
    except:
        print(f'{filename} error, could not convert string to float')
        return None, None, None


    # Read the grid size and spacing
    grid_info = []
    for i in range(3, 6):
        parts = lines[i].split()
        grid_info.append((int(parts[0]), np.array([float(x) for x in parts[1:4]])))


    # extra step
    voxel_x = np.array([float(x) for x in lines[3].split()[1:]])
    voxel_y = np.array([float(y) for y in lines[4].split()[1:]])
    voxel_z = np.array([float(z) for z in lines[5].split()[1:]])

    # intergrate 3*3 step matrix
    voxel_matrix = np.stack([voxel_x, voxel_y, voxel_z])

    grid_data = []
    for line in lines[6+atom_count:]:
        grid_data.extend((float(value) for value in line.split()))
    grid_data = np.array(grid_data)

    if len(grid_data) != int(total_points):
        print(f'{filename} error, Not equeal')
        return None, None, None
    #assert len(grid_data) == int(total_points), f'{filename} error'
    # Reshape density data based on grid dimensions
    nx, ny, nz = grid_info[0][0], grid_info[1][0], grid_info[2][0]
    grid_data = grid_data.reshape((nx, ny, nz))

    return origin, voxel_matrix, grid_data

#def sparse_desity_and_norm_by_17915(voxel_data, origin, save_path, fixed_origin=(0, 0, 0)):
def sparse_desity_and_norm_by_17915(data, save_path):
    origin, voxel_matrix, grid_data = data[0], data[1], data[2]
    num_points = 17915

    points = []
    values = []

    nx, ny, nz = grid_data.shape

    # generate point clound
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                point = origin + i * voxel_matrix[0] + j * voxel_matrix[1] + k * voxel_matrix[2]
                points.append(point)
                values.append(grid_data[i, j, k])

    points = np.array(points)
    values = np.array(values)

    sorted_indices = np.argsort(values)[-num_points:]
    selected_points = points[sorted_indices, :]
    selected_values = values[sorted_indices]

    point_info = {'values': selected_values, 'data': selected_points}
    with open(save_path, 'wb') as f:
        pickle.dump(point_info, f)

'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(selected_points[:, 0], selected_points[:, 1], selected_points[:, 2], c=selected_values[:,], cmap='viridis', s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Electron Density')
    plt.show()
'''

def cal_sparse_desity(dir_data):
    id, dir = dir_data[0], dir_data[1]
    #print(f'processing {id}')
    if dir == 'dude':
        last_path = '/'.join(id.split('/')[2:-1])
    else:
        pass
        last_path = '/'.join(id.split('/')[2:-1])

    pocket_name = id.split('/')[-1]

    density_path = os.path.join(density_fold, last_path, pocket_name,'density.cub')
    origin, voxel_matrix, grid_data = read_cub_file(density_path)


    if origin is None and voxel_matrix is None and grid_data is None:
        print(f"{id} is error")
        os.makedirs(os.path.join('error', last_path, pocket_name), exist_ok=True)
    else:
        data = [origin, voxel_matrix, grid_data]

        save_path = os.path.join(output_point_fold, last_path)
        os.makedirs(save_path, exist_ok=True)
        if os.path.exists(os.path.join(save_path, pocket_name)):
            x = 1
        else:
            sparse_desity_and_norm_by_17915(data, save_path=os.path.join(save_path, pocket_name))



def density2point(input_fold, output_fold):
    dirs_list = os.listdir(input_fold)
    for dir in dirs_list:
        dirs = os.path.join(input_fold, dir)
        subdir_list = os.listdir(dirs)
        pock_list = []
        if dir == 'dude':
            for subsubdirs in subdir_list:
                pock_dirs = os.listdir(os.path.join(dirs, subsubdirs))
                pock_list.append(os.path.join(dirs, subsubdirs))
        if dir != 'dude':
            for subsubdirs in subdir_list:
                pock_dirs = os.listdir(os.path.join(dirs, subsubdirs))
                for pock_id in pock_dirs:
                    pock_list.append(os.path.join(dirs, subsubdirs, pock_id))
        #if dir == 'TrainVal':
        #    pock_dirs = os.listdir(dirs)
        #    for pock_id in pock_dirs:
        #        pock_list.append(os.path.join(dirs, pock_id))
        dir_ = [[data_path, dir] for data_path in pock_list]
        for dir in tqdm(dir_,desc="Processing..."):
            cal_sparse_desity(dir)
        #pool = multiprocessing.Pool(30)
        #pool.starmap(cal_sparse_desity, zip(dir_))
        #pool.close()
        #pool.join()



if __name__ == "__main__":
    data_fold = 'xtb_density/DPI_molden'
    output_fold = 'DPI_density'

    # STEP1 screen mol in calculated density
    #screen_density_cub(data_fold, output_fold)

    density_fold = 'molden/DPI_other_ligands_xyz'
    output_point_fold = 'DPI_point_cloud_17915'
    # STEP2 density.cub to cloud point
    density2point(density_fold, output_point_fold)


