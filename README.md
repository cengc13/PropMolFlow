# PropMolFlow: Property-guided Molecule Generation with Geometry-Complete Flow Matching
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=flat)](https://arxiv.org/abs/2505.21469)

Code for the paper: [*"PropMolFlow: Property-guided Molecule Generation with Geometry-Complete Flow Matching"*](https://arxiv.org/abs/2505.21469)

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
wget https://zenodo.org/records/15700961/files/all_fixed_gdb9.zip
unzip all_fixed_gdb9.zip
rm all_fixed_gdb9.zip
```
After downloading, move the **all_fixed_gdb9.sdf** file to the `data/qm9_raw/` directory:
```bash
mv all_fixed_gdb9.sdf data/qm9_raw/
```

### CSV File for Properties
As for csv file contains properties values, it is provided in `data/qm9_raw` directory. 

## Download Checkpoints

### PropMolFlow Checkpoints
**Note:** We provide **12 model checkpoints** with the **lowest Mean Absolute Error (MAE) values** for **6 molecular properties**,  
covering both **in-distribution** and **out-of-distribution** tasks. 

To download and extract them, run:
```bash
wget https://zenodo.org/records/15700961/files/checkpoints.tar
tar -xvf checkpoints.tar
rm checkpoints.tar
```
Details on how properties are incorporated into the model (e.g., whether Gaussian expansion was used)
can be found in the following files:
* model_result_analysis/in_distribution_best_combo.csv 
* model_result_analysis/out_of_distribution_best_combo.csv

**Note**: We also provide **all 180 model checkpoints**, representing various combinations of
**property handling methods** and **Gaussian expansion settings**. 

The full benchmark results can be found in:
* model_result_analysis/in_distribution_full_results.csv
* model_result_analysis/out_of_distribution_full_results.csv

To download the complete set of checkpoints, run:
```bash
wget https://zenodo.org/records/15700961/files/with_gaussian.tar
tar -xvf with_gaussian.tar
rm with_gaussian.tar

wget https://zenodo.org/records/15700961/files/without_gaussian.tar
tar -xvf without_gaussian.tar
rm without_gaussian.tar
```

### Regressor Checkpoints
**Note:** PropMolFlow does **not** use EGNN for molecular property prediction.  
Instead, we employ **Geometric Vector Perceptrons (GVP)** as the backbone GNN model of PropMolFlow for the property regressor.

The pre-trained checkpoints are already included in the repository under:  
`propmolflow/property_regressor/model_output`  
There is **no need to download** them separately.

## Demo
### Sampling (Generate new property-conditional 3D molecules)

#### In-distribution sampling (samling with many properties values while these values are sampled based on properties distribution)
```bash
# alpha
python sample_condition.py --model_checkpoint "checkpoints/in-distribution/alpha/epoch=1845-step=721785.ckpt" --n_mols 10000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "sum" --multilple_values_file "in_distribution_sampling/train_half_sampled_values_alpha.npy" --number_of_atoms "in_distribution_sampling/train_half_sampled_no_atoms_alpha.npy" --property_name "alpha" --normalization_file_path "data/qm9/train_data_property_normalization.pt" --output_file "sample_result/alpha.sdf" --analyze

# cv
python sample_condition.py --model_checkpoint "checkpoints/in-distribution/cv/epoch=1735-step=678541.ckpt" --n_mols 10000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "sum" --multilple_values_file "in_distribution_sampling/train_half_sampled_values_cv.npy" --number_of_atoms "in_distribution_sampling/train_half_sampled_no_atoms_cv.npy" --property_name "cv" --normalization_file_path "data/qm9/train_data_property_normalization.pt" --output_file "sample_result/cv.sdf" --analyze

# gap
python sample_condition.py --model_checkpoint "checkpoints/in-distribution/gap/epoch=1867-step=730387.ckpt" --n_mols 10000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "concatenate_sum" --multilple_values_file "in_distribution_sampling/train_half_sampled_values_gap.npy" --number_of_atoms "in_distribution_sampling/train_half_sampled_no_atoms_gap.npy" --property_name "gap" --normalization_file_path "data/qm9/train_data_property_normalization.pt" --output_file "sample_result/gap.sdf" --analyze

# homo
python sample_condition.py --model_checkpoint "checkpoints/in-distribution/homo/epoch=1927-step=753691.ckpt" --n_mols 10000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "concatenate" --multilple_values_file "in_distribution_sampling/train_half_sampled_values_homo.npy" --number_of_atoms "in_distribution_sampling/train_half_sampled_no_atoms_homo.npy" --property_name "homo" --normalization_file_path "data/qm9/train_data_property_normalization.pt" --output_file "sample_result/homo.sdf" --analyze

# lumo
python sample_condition.py --model_checkpoint "checkpoints/in-distribution/lumo/epoch=1930-step=754942.ckpt" --n_mols 10000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "concatenate_sum" --multilple_values_file "in_distribution_sampling/train_half_sampled_values_lumo.npy" --number_of_atoms "in_distribution_sampling/train_half_sampled_no_atoms_lumo.npy" --property_name "lumo" --normalization_file_path "data/qm9/train_data_property_normalization.pt" --output_file "sample_result/lumo.sdf" --analyze

# mu
python sample_condition.py --model_checkpoint "checkpoints/in-distribution/mu/epoch=1868-step=730778.ckpt" --n_mols 10000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "multiply" --multilple_values_file "in_distribution_sampling/train_half_sampled_values_mu.npy" --number_of_atoms "in_distribution_sampling/train_half_sampled_no_atoms_mu.npy" --property_name "mu" --normalization_file_path "data/qm9/train_data_property_normalization.pt" --output_file "sample_result/mu.sdf" --analyze
```

#### Out-of-distribution sampling (sampling by giving a certain value of property)
```bash
# alpha
python sample_condition.py --model_checkpoint "checkpoints/out-of-distribution/alpha/epoch=1845-step=721785.ckpt" --n_mols 1000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "sum" --properties_for_sampling 93.3 --number_of_atoms "out_of_distribution_sampling/ood_n_atoms_alpha_up.npy" --property_name "alpha" --normalization_file_path "data/qm9/train_data_property_normalization.pt" --output_file "sample_result/alpha.sdf" --analyze

# cv
python sample_condition.py --model_checkpoint "checkpoints/out-of-distribution/cv/epoch=1735-step=678541.ckpt" --n_mols 1000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "sum" --properties_for_sampling 41.2 --number_of_atoms "out_of_distribution_sampling/ood_n_atoms_cv_up.npy" --property_name "cv" --normalization_file_path "data/qm9/train_data_property_normalization.pt" --output_file "sample_result/cv.sdf" --analyze

# gap
python sample_condition.py --model_checkpoint "checkpoints/out-of-distribution/gap/epoch=1637-step=640223.ckpt" --n_mols 1000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "concatenate" --properties_for_sampling 0.344 --number_of_atoms "out_of_distribution_sampling/ood_n_atoms_gap_up.npy" --property_name "gap" --normalization_file_path "data/qm9/train_data_property_normalization.pt" --output_file "sample_result/gap.sdf" --analyze

# homo
python sample_condition.py --model_checkpoint "checkpoints/out-of-distribution/homo/epoch=1468-step=574066.ckpt" --n_mols 1000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "concatenate_sum" --properties_for_sampling=-0.1757 --number_of_atoms "out_of_distribution_sampling/ood_n_atoms_homo_up.npy" --property_name "homo" --normalization_file_path "data/qm9/train_data_property_normalization.pt" --output_file "sample_result/homo.sdf" --analyze

# lumo
python sample_condition.py --model_checkpoint "checkpoints/out-of-distribution/lumo/epoch=1865-step=729605.ckpt" --n_mols 1000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "concatenate_multiply" --properties_for_sampling 0.091 --number_of_atoms "out_of_distribution_sampling/ood_n_atoms_lumo_up.npy" --property_name "lumo" --normalization_file_path "data/qm9/train_data_property_normalization.pt" --output_file "sample_result/lumo.sdf" --analyze

# mu
python sample_condition.py --model_checkpoint "checkpoints/out-of-distribution/mu/epoch=1883-step=736409.ckpt" --n_mols 1000 --max_batch_size 128 --n_timesteps 100 --properties_handle_method "sum" --properties_for_sampling 6.74 --number_of_atoms "out_of_distribution_sampling/ood_n_atoms_mu_up.npy" --property_name "mu" --normalization_file_path "data/qm9/train_data_property_normalization.pt" --output_file "sample_result/mu.sdf" --analyze
```

#### Sampling Arguments Explanation

- **model_checkpoint**: Path to the trained model checkpoint.
- **n_mols**: Number of molecules to generate.
- **max_batch_size**: Batch size used during sampling.
- **n_timesteps**: Number of integration steps for the ODE solver.
- **properties_handle_method**: Method for incorporating property values into the model.  
  Choices: `'concatenate'`, `'sum'`, `'multiply'`, `'concatenate_sum'`, `'concatenate_multiply'`.
- **number_of_atoms**: Since property values are often correlated with the number of atoms  
  (e.g., molecules with higher heat capacity often contain more atoms),  
  we will explain how this is handled in a following section.
- **property_name**: Property to condition on.  
  Choices: `'alpha'`, `'cv'`, `'gap'`, `'homo'`, `'lumo'`, `'gap'`.  
  These six properties are selected to allow fair comparison with previous works.
- **multiple_values_file**: Path to a `.npy` file containing multiple property values  
  for sampling in **in-distribution** tasks. The model will generate multiple molecules conditioned on diverse values.
- **properties_for_sampling**: A single target value to use for **out-of-distribution** sampling.  
  The model will generate multiple molecules conditioned on this single value.
- **normalization_file_path**: Path to the normalization statistics file.
- **output_file**: Output directory path to save generated molecules.
- **analyze**: Whether to analyze the validity and stability of the generated molecules.


#### Structure-Property Correlation Generation 
This section explains how to generate the files required for the **number_of_atoms** and **multiple_values_file** arguments.

To generate these files, run:
```bash
python str2prop_sampler.py
```
If you want to customize the number of molecular properties generated or change the output path,
please refer to the configurable arguments inside **str2prop_sampler.py**. 

### Prediction and Evaluation
After sampling, we use a pre-trained regressor to predict molecular properties and evaluate the results  
by comparing the **MAE** between the input values and the predicted values.  
The prediction results will be saved in the `prediction_result/` directory.

To run **in-distribution** prediction and calculate MAE, use the following command:
```bash
# this is for in-distribution prediction and calculate MAE values
python propmolflow/property_regressor/test_regressor.py --checkpoint=propmolflow/property_regressor/model_output/alpha/checkpoints/gvp-regressor-epoch=358-val_loss=0.0061.ckpt --config=propmolflow/property_regressor/configs/test.yaml --input=sample_result/example_alpha.sdf --output=example_case.pt --properties_values=in_distribution_sampling/train_half_sampled_no_atoms_alpha.npy --property_name=alpha
```
**Note**: For **out-of-distribution** prediction, we need to give a certain value for `--properties_values` argument:
 ```bash 
python propmolflow/property_regressor/test_regressor.py --checkpoint=propmolflow/property_regressor/model_output/alpha/checkpoints/gvp-regressor-epoch=358-val_loss=0.0061.ckpt --config=propmolflow/property_regressor/configs/test.yaml --input=sample_result/example_alpha.sdf --output=example_case.pt --properties_values=93.3 --property_name=alpha
```
You can find values we used for **out-of-distribution** task in `out_of_distribution_sampling/chosen_target_values.json` for different properties. This json file can be generated by running **str2prop_sampler.py**.


### Training 
**Note**: For training the PropMolFlow model, we need to run following command to process the qm9 dataset first:
```bash
python process_qm9_cond.py --config=configs/with_gaussian/alpha_concatenate_multiply.yaml
```

Then run the **train.py** script. You can either pass a config file, or pass a trained model checkpoint for resuming.
```python
# training from scratch
python train.py --config=configs/without_gaussian/cv_concatenate_sum.yaml 

# continue training from checkpoints
python train.py --resume=checkpoints/without_gaussian/
```

## Acknowledgements
PropMolFlow builds upon the source code from [FlowMol](https://github.com/Dunni3/FlowMol). We sincerely thank the authors of FlowMol for their excellent work and contributions.