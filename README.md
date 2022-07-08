# AISIP-MBDA
Model-based data augmentation for MRI 

## Quick References

### Related works
* Functional Magnetic Resonance Imaging data augmentation through conditional ICA: https://arxiv.org/abs/2107.06104
* Evaluation of Data Augmentation of MR Images for Deep Learning: https://lup.lub.lu.se/luur/download?func=downloadFile&recordOId=8952747&fileOId=8952748
* Data augmentation for deep learning based accelerated MRI reconstruction with limited data: https://arxiv.org/abs/2106.14947
* FMRI data augmentation via synthesis: https://arxiv.org/abs/2106.14947
* TorchIO Paper: https://www.sciencedirect.com/science/article/pii/S0169260721003102

### Datasets
* HCP900: https://www.humanconnectome.org/study/hcp-young-adult/document/900-subjects-data-release
* Neurovault: https://neurovault.org/

### Code and Librarier
* TorchIO: https://torchio.readthedocs.io/
* Nilearn: https://nilearn.github.io/stable/index.html



## Installation 
To install the code as a package, run the following code in your home directory: 

```
pip install poetry
poetry install
```

### Execution pipeline

#### Data loading and preparation

1. To upload the data from neurovelt, run the notebooks/download_hcp.ipynb notebook.
The script will download 3D temporal statistics of a 4D fMRI data into your provided directory. It will also upload and resample to the size of your 3D data matrixes Z, mask, and Z-inverse that wil be then used to project 3D matrices into DiFuMo space. 


2. To then tune augmentation parameters, use notebooks/torchio_data_augmentation_widgets.ipynb notebook. The notebook will provide you a set of widgets that show projected into the glass brain original image and either augmented 3D image or the difference between original and augmented image.

# DiFuMo extraction with and without augmentations

Once you found a set of augmentation and its parameters that is satisfactory, fix their 

We will first store the 
To run the extraction of DiFuMo vectors without and with augmentations, run 

```
python ai4sipmbda/fixed_augmentation_generation.py --base_dataset_path "../../Data/neurovault/neurovault/collection_4337" --difumo_maps_path "../../Data/hcp900_difumo_matrices/" --save_path "../../Data/HCP_difumo" --num_samples 15
```
