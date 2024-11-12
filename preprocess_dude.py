import os

# 设置主文件夹路径
main_folder = 'DPI_data_for_xyz/ligands/dude'

# 遍历主文件夹中的所有子文件夹
for folder_name in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder_name)

    # 检查是否为文件夹
    if os.path.isdir(folder_path):
        # 设置要保留的文件路径（目标文件）
        target_file = os.path.join(folder_path, 'crystal_ligand.mol2')

        # 如果目标文件存在
        if os.path.exists(target_file):
            # 重命名目标文件为 '{folder_name}_ligand.mol2'
            new_file_name = f'{folder_name}_ligand.mol2'
            new_file_path = os.path.join(folder_path, new_file_name)
            os.rename(target_file, new_file_path)
            print(f'Renamed {target_file} to {new_file_path}')

        # 删除文件夹中除重命名后的目标文件以外的所有文件
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name != new_file_name:  # 排除已重命名的文件
                os.remove(file_path)
                print(f'Deleted: {file_path}')

print("Process completed.")
