# Supporting Information

This repository contains the Supporting Information code for: 
***Sanocki, M., & Zavadlav, J. (2025). Generalization of Long-Range Machine Learning Potentials in Complex Chemical Spaces. ArXiv. https://arxiv.org/abs/2512.10989***



## Summary

This repository contains code grouped under:
  - chemtrain/   — code for training and evaluation of models that depend on the chemtrain code. For installation details, see: https://github.com/tummfm/chemtrain
  - les/         — code for training and evaluation of models that depend on LES and MACE code. For installation details, see: https://github.com/ChengUCB/les/tree/main
  - Dataset_creation/ - code for Dataset and biased split creation 

Requirements for chemtrain and les are available in their corresponding directories.

Information on the datasets utilized in this study is available at:
  - QMOF https://github.com/Andrew-S-Rosen/QMOF
  - ODAC25 https://huggingface.co/facebook/ODAC25
  - OMOL25 https://huggingface.co/facebook/OMol25

## Training

Each split trained on the QMOF dataset has seperate training split, whereas the split for models trained on OMOL25 and ODAC25 has to be specified using the command line (path for maxsep and cluster splits or seed for random, for size split --sl True). 

## chemtrain 

### ODAC/OMOL
Chemtrain-based models trained on ODAC25 or OMol25 are launched as:

python SCRIPT.py GPU_ID --traj PATH_TO_TRAJ [split options]


Split strategies

Cluster / Max Separation (external indices)

--train_indices path/to/train_indices.txt
--test_indices  path/to/test_indices.txt


Example:

python allegro_efa_odac.py "1" \
  --traj path/to/dataset.traj \
  --train_indices path/to/train_indices.txt \
  --test_indices path/to/test_indices.txt


Random split: python allegro_efa_odac.py "1" --traj path/to/dataset.traj --seed 3


SL split (small/large): python allegro_efa_odac.py "1" --traj path/to/dataset.traj --seed 3 --sl True

### QMOF

For QMOF, dataset splitting is defined inside each training script.

Standard preprocessing uses:

preprocess_mof_data(data_path, split_method="cluster", val_ratio=0.1, seed=3)


Supported split_method values: cluster, maxsep, None (random fallback)

Cluster and MaxSep require:

train_refcodes_<split_method>.csv
test_refcodes_<split_method>.csv

SL (small/large) experiments use a separate preprocessing function:

preprocess_mof_data(data_path, train_cutoff=100)


Models are launched as:

python SCRIPT.py GPU_IDS



## les

LES-based models are trained using the provided fit.sh script, which wraps run_train.py and specifies all model and training parameters.



Datasets are provided as .extxyz files:

--train_file

--valid_file

--test_file


All model architecture, optimization, and training hyperparameters (MACELES, channels, cutoff, interactions, EMA, SWA, etc.) are defined directly in fit.sh.

Each run produces standard LES/MACE output logs and checkpoints in the working directory.


