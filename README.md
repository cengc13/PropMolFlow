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

## Datasets 


## Acknowledgements
PropMolFlow builds upon the source code from [FlowMol](https://github.com/Dunni3/FlowMol). We sincerely thank the authors of FlowMol for their excellent work and contributions.