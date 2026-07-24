# EvoStruct-GAT: Protein-Protein Interface Residue Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

Official repository for the paper **"Protein–Protein Interface Residue Prediction by Integrating Protein Language Representations and Structural Graph Attention"**.

EvoStruct-GAT is a multimodal graph attention model designed for the accurate residue-level prediction of protein-protein interaction (PPI) interfaces. The model represents each target-chain residue as a graph node, integrates residue-level representations from the ESM-2 protein language model with DSSP-derived structural features, and learns spatial neighborhood information through distance-aware graph attention.

## 📊 Dataset
The model is trained and evaluated on the PDBbind v2020.R1 protein-protein complex dataset. 
* **Training set:** 602 complexes
* **Validation set:** 174 complexes
* **Test set:** 175 complexes
* Data splitting details are provided in `dataset_split_record_regenerated.csv`. 
* Residue-level interface labels (5 Å distance threshold) are stored in the `labels/` directory.

## 📂 Project Structure

```text
EvoStruct-GAT/
├── 00_full_pipeline.py           # Raw PDB cleaning and initial metadata generation
├── 01_filter_and_copy_pdb.py     # Filters complexes by resolution (≤3.0Å) and chain length
├── 02_trim_extra_chains.py       # Removes non-interacting chains from complexes
├── 03_split_dataset_final.py     # Dataset splitting script
├── 04_extract_global_sequences.py# Sequence extraction for CD-HIT redundancy removal
├── 06_generate_interface_labels.py# Calculates interface labels (5Å cross-chain threshold)
├── 07_build_graph.py             # Constructs PyG spatial graphs with ESM & DSSP features
├── 08_dataset.py                 # PyTorch Geometric custom Dataset loader
├── 09_model.py                   # EvoStruct-GAT model architecture
├── 10_train.py                   # Main training script (w/ Focal Loss & Undersampling)
├── 11_benchmark.py               # Benchmark comparisons vs Feature-MLP and Standard GCN
├── 12_eval_best_model.py         # Test set evaluation, optimal thresholding, and plotting
├── 13_predict.py                 # CLI tool for predicting hotspots on novel PDB chains
├── dataset_split_record_regenerated.csv # Official dataset split lists
├── example_data/                 # Sample dataset containing PDB, CSV features, and PYG graphs for quick tests
├── labels/                       # Pre-calculated residue labels for all complexes
├── model_save/                   # Contains trained weights (best_model.pth) and academic plots
└── ablation_study/                      # Ablation study scripts and variants