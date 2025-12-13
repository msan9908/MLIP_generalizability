# Supporting Information

This repository contains the Supporting Information code for: XXXXXXXX



## Summary

This repository contains code grouped under:
  - chemtrain/   — code for training and evaluation of models that depends on chemtrain package. For installation details, see: https://github.com/tummfm/chemtrain
  - les/         — code for training and evaluation of models that depends on LES and MACE code. For installation details see: https://github.com/ChengUCB/les/tree/main
  - Dataset_creation/ - code for Dataset and biased split creation 


Information on Datasets utilized in this study is availble at:
  - QMOF https://github.com/Andrew-S-Rosen/QMOF
  - ODAC25 https://huggingface.co/facebook/ODAC25
  - OMOL25 https://huggingface.co/facebook/OMol25

## Training

Each split trained on the QMOF dataset has seperate training split, wheraes split for models trined on OMOL25 and ODAC25 has to be specified using command line (path fro maxsep and cluster splits or seed for random, for size split --sl True)


