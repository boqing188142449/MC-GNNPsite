# Introduction
MC-GNNPsite is a multi-channel graph neural network (MC-GNNPsite) designed for predicting protein binding sites, aiming to enhance the accuracy of sequence-based predictions.

# System requirement
MC-GNNPsite was developed on a Linux environment with CUDA 12.6.

python  3.7.12  
numpy  1.21.2  
pandas  1.3.5  
torch  1.13.1  
biopython  1.79  
sentencepiece 0.2.0  
transformers 4.30.2

# Install and set up
**1.** Clone this repository

`git clone https://github.com/boqing188142449/MC-GNNPsite.git` or download the code in ZIP archive. The latest version is `MC-GNNPsite-master`.  

**2.** Convert the protein txt file into  protein folders:

`python Utils/fastaToFiles.py txt_folder Protein_folder `

**3.** Download the [ESMFold](https://github.com/facebookresearch/esm) and [ESM-2](https://github.com/facebookresearch/esm) model and install according to the official tutorials：

```
pip install fair-esm  
pip install git+https://github.com/facebookresearch/esm.git 
pip install "fair-esm[esmfold]"
# OpenFold and its remaining dependency
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
```

Predict protein structures: `python ./Utils/esmfold_to_pdb.py Protein folder`  

Extract ESM-2 embeddings: `python ./Utils/esm_to_embed.py Protein folder ` 

**4.** Download the pre-trained ProtT5-XL-UniRef50 model in [here](https://github.com/agemagician/ProtTrans). The downloaded model is stored in `.\Utils\model_cache`.  Extracting language model features ：

```
python ./Utils/prott5_to_embed.py Protein_folder
```
**5.** Add permission to execute for [DSSP](https://github.com/PDB-REDO/dssp)  and extract dssp:

```
python ./Utils/generate_dssp.py Protein_folder
python ./Utils/dssp_to_npy.py Protein_folder
```
**6.** The adjacency matrix is obtained according to different threshold values:

```
python ./Utils/generate_dssp.py Protein_folder
python ./Utils/dssp_to_npy.py Protein_folder
```

**7.** Save the [Kidera Factors](https://github.com/vadimnazarov/kidera-atchley?tab=readme-ov-file) data as `kidera.csv` and store it in `.\Utils\kidera.csv`. Extract the Kidera factor features for each amino acid.

`python ./Utils/generate_kidera.py Protein_folder `

**8.** Merge all the features:

```
python .Utils/merge_feature.py
```
# Run MC-GNNPsite for train
Run the following command to start training:
```
python ./train.py
```
# Run MC-GNNPsite for test

Run the following command to start testing:

```
python ./test.py
```

# Datasets and models

The datasets used in this study are stored in `./datasets/`  
The trained MC-GNNPsite models can be found in `https://drive.google.com/drive/folders/1cv2mJY6Dnv3tArQ2r4Y-aUHR6Qk-fGtl?usp=sharing`

# Citation and contact
```
@article{
}
```
 
