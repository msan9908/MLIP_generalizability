# Supporting Information

This repository contains the Supporting Information code for:

> **Sanocki, M., & Zavadlav, J. (2025)**  
> *Generalization of Long-Range Machine Learning Potentials in Complex Chemical Spaces*  
> https://arxiv.org/abs/2512.10989

---

## 📦 Repository Structure

The repository is organized into the following components:

- `chemtrain/` – Training and evaluation code based on **chemtrain**  
  → Installation: https://github.com/tummfm/chemtrain  

- `les/` – Training and evaluation code based on **LES** and **MACE**  
  → Installation: https://github.com/ChengUCB/les/tree/main  

- `Dataset_creation/` – Scripts for dataset construction and biased split generation  

Each submodule contains its own dependency requirements.

---

## 📊 Datasets

The datasets used in this study are available at:

- **QMOF**  
  https://github.com/Andrew-S-Rosen/QMOF  

- **ODAC25**  
  https://huggingface.co/facebook/ODAC25  

- **OMol25**  
  https://huggingface.co/facebook/OMol25  

---

## 🏋️ Training Overview

- **QMOF models**: use predefined dataset splits (handled internally in scripts)  
- **OMol25 / ODAC25 models**: require split specification via command-line arguments  

Split types include:

- Cluster split  
- Maximum separation (maxsep)  
- Random split  
- Small/Large (SL) split  

---

# 🔬 chemtrain

## ODAC / OMol Training

Run chemtrain-based models as:

```bash
python SCRIPT.py GPU_ID --traj PATH_TO_TRAJ [split options]
```

### Split Strategies

#### Cluster / Max Separation (external indices)

```bash
--train_indices path/to/train_indices.txt
--test_indices  path/to/test_indices.txt
```

#### Example

```bash
python allegro_efa_odac.py "1" \
  --traj path/to/dataset.traj \
  --train_indices path/to/train_indices.txt \
  --test_indices path/to/test_indices.txt
```

#### Random Split

```bash
python allegro_efa_odac.py "1" \
  --traj path/to/dataset.traj \
  --seed 3
```

#### Small/Large (SL) Split

```bash
python allegro_efa_odac.py "1" \
  --traj path/to/dataset.traj \
  --seed 3 \
  --sl True
```

---

## QMOF Training

For QMOF, dataset splitting is handled inside each training script.

### Standard Preprocessing

```python
preprocess_mof_data(
    data_path,
    split_method="cluster",
    val_ratio=0.1,
    seed=3
)
```

Supported `split_method` values:

- `cluster`
- `maxsep`
- `None` (fallback to random split)

### Required Files for Cluster / MaxSep

```
train_refcodes_<split_method>.csv
test_refcodes_<split_method>.csv
```

### Small/Large (SL) Preprocessing

```python
preprocess_mof_data(
    data_path,
    train_cutoff=100
)
```

### Running Models

```bash
python SCRIPT.py GPU_IDS
```

---

# ⚙️ les

LES-based models are trained using the provided `fit.sh` script.

### Training Entry Point

```bash
bash fit.sh
```

This script wraps:

```bash
run_train.py
```

and defines all model and training parameters.

---

## 📂 Dataset Format

Datasets must be provided as `.extxyz` files:

```bash
--train_file path/to/train.extxyz
--valid_file path/to/valid.extxyz
--test_file  path/to/test.extxyz
```

---

## ⚙️ Configuration

All key parameters are defined directly in `fit.sh`, including:

- Model architecture (MACELES)
- Number of channels
- Cutoff radius
- Number of interactions
- Optimization settings
- EMA / SWA

---

## 📈 Outputs

Each training run produces:

- Standard LES/MACE logs  
- Model checkpoints  
- Outputs saved in the working directory  
