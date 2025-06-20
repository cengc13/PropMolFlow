# PropMolFlow: Property-guided Molecule Generation with Geometry-Complete Flow Matching
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=flat)](https://arxiv.org/abs/2505.21469)

![Image](figure/overview_page.jpg)

## Environment Setup
1. Create a conda environment with python 3.10: `conda create -n propmolflow python=3.10`
2. Activate the environment: `conda activate propmolflow`
3. Install the packages required: 
```python
conda install mamba
mamba install pytorch=2.2.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
mamba install pytorch-cluster=1.6.3 pytorch-scatter=2.1.2=py310_torch_2.2.0_cu121 -c pyg -y
mamba install -c dglteam/label/cu121 dgl=2.0.0.cu121=py310_0 -y
mamba install -c conda-forge pytorch-lightning=2.1.3=pyhd8ed1ab_0 -y
mamba install -c conda-forge pystow einops -y

pip install rdkit==2023.9.4
pip install numpy==1.26.3
pip install wandb useful_rdkit_utils py3Dmol --no-input
pip install -e .
```

## Download Datasets 

### QM9 SDF File
We provide a corrected version of the QM9 SDF file originally from [DeepChem](https://github.com/deepchem/deepchem), fixing issues such as **invalid bond orders** and **non-zero net charges** in approximately **30,000 molecules**.

To download the fixed SDF file, run:
```bash
wget https://zenodo.org/uploads/15700961/files/all_fixed_gdb9.zip
unzip all_fixed_gdb9.zip
rm all_fixed_gdb9.zip
```
After downloading, move the all_fixed_gdb9.sdf file to the `data/qm9_raw/` directory:
```bash
mv all_fixed_gdb9.sdf data/qm9_raw/
```

### CSV File for Properties
As for csv file contains properties values, it is provided in `data/qm9_raw` directory. 

## Download Checkpoints

### PropMolFlow Checkpoints
**Note**: This containts 60 checkpoints for 6 properties with 5 properties handling method in with/without guassuain expansions.
```bash
wget https://zenodo.org/record/13375913/files/GCDM_Checkpoints.tar.gz
tar -xzf GCDM_Checkpoints.tar.gz
rm GCDM_Checkpoints.tar.gz
```

### Regressor Checkpoints
**Note:** PropMolFlow does **not** use EGNN for molecular property prediction.  
Instead, we employ **Geometric Vector Perceptrons (GVP)** as the backbone GNN model of PropMolFlow for the property regressor.

The pre-trained checkpoints are already included in the repository under:  
`propmolflow/property_regressor/model_output`  
There is **no need to download** them separately.

## Demo
### Sampling (Generate new property-conditional 3D molecules)

#### In-distrubtion sampling (sampling by giving one certain value)
```bash
# alpha
python sample_condition.py --model_checkpoint "$MODEL_CHECKPOINT" --n_mols 10000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "concatenate_sum" --properties_for_sampling 41.2 --number_of_atoms "/blue/mingjieliu/jiruijin/diffusion/FlowMol_active/sampling_result/ood_n_atoms_cv_up.npy" \
  --property_name "cv" \
  --normalization_file_path "/blue/mingjieliu/jiruijin/diffusion/FlowMol/data/qm9/train_data_property_normalization.pt" \
  --output_file "$OUT_DIR/cv.sdf" \
  --analyze
```

### Training 
Run the **train.py** script. You can either pass a config file, or pass a trained model checkpoint for resuming.
```python
# training from scratch
python train.py --config=configs/without_gaussian/cv_concatenate_sum.yaml 

# continue training from checkpoints
python train.py --resume=checkpoints/without_gaussian/
```

## Acknowledgements
PropMolFlow builds upon the source code from [FlowMol](https://github.com/Dunni3/FlowMol). We sincerely thank the authors of FlowMol for their excellent work and contributions.