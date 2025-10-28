## Introduction
The **E-CloudBind** project consists of a **preprocessing part** and a **demo testing part**. Its structure is as follows:

```bash
├─ E-CloudBind/
│   ├─ E-CloudBind/   # Demo execution code; dataset assembly; model architecture and implementation
│   ├─ Preprocess/    # Procedures for ligand electron-density and pocket pseudo–electron-density generation
│   └─ README.md      # Project documentation (usage, environment, quick start)
```
## Installation
```bash
# Packages version
torch==1.12.1  
dgl-cu113==0.9.1.post1  
chemprop==1.7.1  
descriptastorus==2.8.0  
rdkit==2024.3.3  
scipy==1.9.0  
pandas==2.0.3  
scikit-learn==1.2.2  
matplotlib==3.7.5  
seaborn==0.13.2  
tqdm==4.67.1
```

```bash
conda create -n ecloudbind python=3.8
conda activate ecloudbind
cd E-CloudBind/E-CloudBind && pip install -r requirements.txt
```
> *Note: Install xTB from [https://github.com/grimme-lab/xtb](https://github.com/grimme-lab/xtb) and install Multiwfn from [http://sobereva.com/multiwfn/](http://sobereva.com/multiwfn/)*


## Preprocessing
Project structure of the preprocessing part:
```bash
├─ Preprocess/
│   ├─ 1_extra_coord_PocAndMol.py
│   ├─ 2_Mol_xyz2molden.sh
│   ├─ 3_Mol_molden2density.sh
│   ├─ 4_Mol_cub2voxel.py
│   ├─ 5_Mol_cluster_point.py
│   ├─ 6_Poc_extra_pock_point.py
│   ├─ 7_preprocess_complex.py
│   └─ 8_graph_constructor.py
├─ DPI_data/
│   └─ <ID>/
│       ├─ <ID>_protein.pdb
│       └─ <ID>_ligand.mol2    
├─ toy_examples.csv           
```
Before running, the **ligand** and **protein** files need be provided (ligand: `.mol2`; protein: `.pdb`). Using `6upj` as an example, `DPI_data/6upj/` should contain `6upj_protein.pdb` and `6upj_ligand.mol2`. The following steps are then executed in order:
```bash
# Step 1: Export ligand & pocket XYZ
cd Preprocess && python 1_extra_coord_PocAndMol.py

# Step 2: Ligand XYZ → molden.input (depends on semi-empirical xTB)
bash 2_Mol_xyz2molden.sh

# Step 3: molden.input → density.cub (depends on Multiwfn)
bash 3_Mol_molden2density.sh

# Step 4: density.cub → sparse point cloud
python 4_Mol_cub2voxel.py   #  Point cloud filtering

# Step 5: Cluster / balance the point cloud
python 5_Mol_cluster_point.py   # Final ligand electron cloud

# Step 6: Pocket XYZ → pocket point cloud
python 6_Poc_extra_pock_point.py    #  Protein electron-like cloud via van der Waals radii

# Step 7: Build RDKit complex & pocket slice
python 7_preprocess_complex.py      

# Step 8: Construct DGL graph
python 8_graph_constructor.py       # Produces Graph_E_cloudBind-<ID>.dgl

# Step 9: Pack to toy/<id>/ (for demo)
python 9_change_format.py          # Then copy toy dir to E-CloudBind/data
```

After running, you should see the **key files** in these locations:

- `DPI_xyz/<ID>/{<ID>_ligand.xyz, <ID>_pocket.xyz}` – raw coordinates for ligand & pocket  
- `molden/DPI_xyz/<ID>/<ID>_ligand/{molden.input, density.cub, ...}` – xTB/Multiwfn products  
- `point_cloud_17915/<id> & point_cloud_with_clus/<ID>` – ligand density point clouds (raw & balanced)  
- `DPI_pocket_cloud_fixed_cluster_num_multivariate_normal/<id>/<pocket>.pkl` – pocket point cloud  
- `DPI_data/<ID>/{<ID>.rdkit}` – RDKit complex 
- `DPI_data/<ID>/Graph_E_cloudBind-<ID>.dgl` – molecular DGL graph
- `toy/<id>/{Graph_E_cloudBind-<id>.dgl, DPI_complex_pkl/<id>.pkl, point_cloud_with_clus/<id>}` – packaged preprocessing outputs for demo


## E-CloudBind 
Project structure of the E-CloudBind part:
```bash
├─ E-CloudBind/
│   ├─ data/                          # Example/runtime data
│   ├─ weights/                       # Pretrained weights
│   ├─ E-CloudBind_demo.py            # Inference/demo entry: load weights and run predictions on examples
│   ├─ Model.py                       # E-CloudBind model definition
│   ├─ HGC.py                         # Graph components (hierarchical/hybrid aggregation modules)
│   ├─ NIGConv.py                     # Graph convolution operator implementation
│   ├─ CIGConv.py                     # Graph convolution operator implementation
│   ├─ gcn3d.py                       # 3D graph convolution / geometric modeling utilities
│   ├─ model_gcn3d.py                 # Model variant built on 3D GCN
│   ├─ graph_constructor_v2.py        # Dataset
│   ├─ utils.py                       # General utilities (I/O, metrics, logging, helpers)
│   └─ requirements.txt               # Python dependency list
```
We provide an **inference demo** for E-CloudBind together with pretrained weights via the Python script `E-CloudBind_demo.py`. In addition, five preprocessed protein–ligand toy cases from the PDBbind dataset are included. Further examples can be produced using the `Preprocessing` pipeline.

**To run E-CloudBind inference:**
```bash
cd Preprocess && cp -r toy/ ../E-CloudBind/data/   # copy the packaged toy set into E-CloudBind/data
cd ../E-CloudBind && python E-CloudBind_demo.py    # launch the demo (loads weights and runs inference)
```

After running, you will see the model’s performance metrics printed in the console.

```bash
point clound data size is 5
valid_rmse: 0.2779, valid_pr: 0.9712
```