# Installation Instructions

This guide walks you through setting up the **Germinal** environment and installing all necessary dependencies.

---

## 1. Create and Activate Conda Environment

```bash
conda create --name germinal python=3.10
conda activate germinal
```

---

## 2. Install Core Packages

```bash
pip install pandas matplotlib numpy biopython scipy seaborn tqdm ffmpeg py3dmol \
  chex dm-haiku dm-tree joblib ml-collections immutabledict optax cvxopt mdtraj colabfold
```

---

## 3. Install JAX with CUDA Support

```bash
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install -c nvidia cuda-nvcc
conda install -c nvidia cudnn
```

---

## 4. Install ColabDesign and PyRosetta

> **Note:** Make sure you are in the **Germinal root directory** before running this.

```bash
# ColabDesign (editable install)
pip install -e colabdesign

# PyRosetta
pip install pyrosetta-installer
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
```

---

## 5. Install Torch, Chai, and IgLM

```bash
pip install iglm torchvision==0.21.* chai-lab==0.6.1 \
  torch==2.6.* torchaudio==2.6.* torchtyping==0.1.5 torch_geometric==2.6.*
```

**NOTE: ignore colabfold dependency errors

---

## 6. Install Project in Editable Mode

```bash
pip install -e .
```

---

## 7. Ensure Dependency Compatibility

To resolve version mismatches, install the following pinned versions:

```bash
pip install jax==0.5.3
pip install dm-haiku==0.0.13 
pip install hydra-core omegaconf
pip install "jax[cuda12_pip]==0.5.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

