# EvoStruct-GAT: Protein–Protein Interface Residue Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-PyG-3C2179.svg)](https://pyg.org/)

Official code repository for the manuscript:

**Protein–Protein Interface Residue Prediction by Integrating Protein Language Representations and Structural Graph Attention**

EvoStruct-GAT is a multimodal graph-attention framework for residue-level protein–protein interface prediction. The model integrates pretrained **ESM-2** residue representations with **DSSP-derived structural descriptors**, represents each target protein side as a three-dimensional residue graph, and uses distance-aware **GATv2** message passing to rank candidate interface residues.

A key feature of the released workflow is its **partner-independent prediction setting**. The interacting partner is used only during dataset construction to generate reference interface labels. Partner atoms are excluded from target-side DSSP calculation, node features, spatial edges, and graph construction.

---

## Dataset and evaluation protocol

The project was developed using protein–protein complexes from **PDBbind v2020.R1**.

Receptor-side and ligand-side sequences were clustered independently using CD-HIT at 40% sequence identity. A cluster-constrained partitioning strategy was then applied so that complexes sharing either a receptor cluster or a ligand cluster remained in the same dataset partition.

The final benchmark contained:

- **1,390 training complexes**
- **172 validation complexes**
- **177 test complexes**

The primary interface labels were defined using a minimum cross-partner heavy-atom distance of **<5 Å**.

Additional sensitivity experiments were carried out with alternative interface definitions:

- **4.0 Å**
- **4.5 Å**
- **ΔSASA1**

Because the complete PDBbind-derived structural dataset and generated PyG graph collection are large, this repository contains only a **small representative subset of PDB files and graph files**, retained directly in their original pipeline directories. The public repository therefore illustrates the expected input/output formats but does not redistribute the full experimental dataset.

---

## Repository structure

```text
EvoStruct-GAT/
├── README.md
├── .gitignore
│
├── 00_filter_and_copy_pdb.py
├── 00_full_pipeline.py
├── 00_trim_extra_chains.py
├── 01_extract_complex_sequences.py
├── 02_build_complex_cluster_mapping.py
├── 02_complex_chain_processing_QC.py
├── 03_cluster_level_split1.py
├── 04_generate_labels_v3.py
├── 05_build_graph.py
├── 05_make_sensitivity_graphs.py
├── 06_dataset.py
├── 06_dataset_sensitivity.py
├── 07_model.py
├── 08_train.py
├── 08train_sensitivity.py
├── 09_ablation_all_in_one_partner_only_DSSP.py
├── 09_ablation_all_in_one_with_full_control.py
├── 09_eval_best_model.py
├── 09_eval_best_model_sensitivity.py
├── 11_residue_statistics_partner_only_DSSP.py
├── CR1_case_analysis_old_model_topK_fixed.py
├── CR1_reviewer_recall_summary.py
├── internal_model_comparison_5seeds.py
├── internal_model_comparison_val_selection.py
├── qc_generate_final_metadata.py
├── filtered_pdb_metadata_final.csv
├── final_metadata.csv
├── dssp.exe
│
├── 03_chain_trimmed_pdb/
│   ├── 1a4v_complex.pdb
│   ├── 1a22_complex.pdb
│   ├── 1acb_complex.pdb
│   ├── 1ahw_complex.pdb
│   ├── 1axi_complex.pdb
│   └── 1ay7_complex.pdb
│
├── CR1_homology_check/
│   └── ...
│
└── cluster_split_v3_A2/
    ├── 01_complex_cluster_input/
    ├── 02_cdhit/
    ├── 03_complex_cluster_mapping/
    ├── 04_split/
    ├── 05_labels/
    ├── chain_processing_QC/
    │
    ├── partner_only_DSSP/
    │   ├── 06_graphs/
    │   │   ├── train_pyg_3d_graph/
    │   │   ├── val_pyg_3d_graph/
    │   │   └── test_pyg_3d_graph/
    │   ├── 07_model_save/
    │   ├── 09_ablation/
    │   ├── 11_CR1_case_analysis/
    │   └── residue_statistics/
    │
    └── sensitivity_analysis/
        ├── 4A/
        ├── 4p5A/
        │   └── 06_graph/
        │       └── train_pyg_3d_graph/
        ├── deltaSASA1/
        └── sensitivity_graph_generation_summary.csv
```

**Note:** The public graph/data directories contain only selected representative files. They are intentionally incomplete relative to the full experimental dataset.

---

## Pipeline overview

### 1. Structure filtering and chain preprocessing

```text
00_filter_and_copy_pdb.py
00_trim_extra_chains.py
02_complex_chain_processing_QC.py
qc_generate_final_metadata.py
```

These scripts perform initial PDB filtering, chain trimming, chain-level quality control, and metadata generation.

### 2. Sequence extraction and cluster-constrained splitting

```text
01_extract_complex_sequences.py
02_build_complex_cluster_mapping.py
03_cluster_level_split1.py
```

Receptor and ligand sequences are processed independently for clustering. Complexes linked through either receptor-cluster or ligand-cluster membership are assigned to the same data partition to reduce sequence-related information leakage across training, validation, and test subsets.

Split-related files are stored under:

```text
cluster_split_v3_A2/
├── 01_complex_cluster_input/
├── 02_cdhit/
├── 03_complex_cluster_mapping/
└── 04_split/
```

### 3. Interface-label generation

```text
04_generate_labels_v3.py
```

The primary residue-level interface labels are generated using a minimum cross-partner heavy-atom distance of <5 Å.

Generated labels are stored under:

```text
cluster_split_v3_A2/05_labels/
```

### 4. Partner-independent graph construction

```text
05_build_graph.py
```

For each prediction target:

- ESM-2 provides residue-level sequence-context representations.
- DSSP provides secondary-structure and relative solvent-accessibility descriptors.
- residues are represented as graph nodes;
- target-side spatial relationships define graph edges;
- inter-residue distances are encoded as edge attributes.

Partner atoms are not used for target-side graph construction or structural feature calculation.

Generated graphs are stored under:

```text
cluster_split_v3_A2/partner_only_DSSP/06_graphs/
```

Only selected graph files are included in the public repository.

### 5. Dataset loading and model definition

```text
06_dataset.py
07_model.py
```

`06_dataset.py` implements graph-data loading and sampling logic.

`07_model.py` defines the EvoStruct-GAT architecture.

### 6. Training and model evaluation

```text
08_train.py
09_eval_best_model.py
```

Training outputs, evaluation results, and the released trained model checkpoint(s) are stored under:

```text
cluster_split_v3_A2/partner_only_DSSP/07_model_save/
```

This directory may contain `.pt` or `.pth` model-weight files together with evaluation summaries and output files.

### 7. Ablation experiments

```text
09_ablation_all_in_one_partner_only_DSSP.py
09_ablation_all_in_one_with_full_control.py
```

Ablation outputs are stored under:

```text
cluster_split_v3_A2/partner_only_DSSP/09_ablation/
```

### 8. Internal model comparisons

```text
internal_model_comparison_5seeds.py
internal_model_comparison_val_selection.py
```

These scripts implement controlled comparisons between EvoStruct-GAT and simplified internal baseline architectures.

### 9. Interface-definition sensitivity analysis

```text
05_make_sensitivity_graphs.py
06_dataset_sensitivity.py
08train_sensitivity.py
09_eval_best_model_sensitivity.py
```

Alternative interface-definition experiments are organized under:

```text
cluster_split_v3_A2/sensitivity_analysis/
├── 4A/
├── 4p5A/
└── deltaSASA1/
```

Selected sensitivity-analysis graph files are retained directly in their corresponding graph directories.

### 10. CR1 case study and residue statistics

```text
CR1_case_analysis_old_model_topK_fixed.py
CR1_reviewer_recall_summary.py
11_residue_statistics_partner_only_DSSP.py
```

CR1 case-study outputs are stored under:

```text
cluster_split_v3_A2/partner_only_DSSP/11_CR1_case_analysis/
```

Residue-level statistical summaries are stored under:

```text
cluster_split_v3_A2/partner_only_DSSP/residue_statistics/
```

---

## Representative data files

The repository contains a small number of representative processed PDB structures under:

```text
03_chain_trimmed_pdb/
```

Example graph objects are retained directly under the corresponding original graph directories, for example:

```text
cluster_split_v3_A2/partner_only_DSSP/06_graphs/test_pyg_3d_graph/
```

and sensitivity-analysis graph directories such as:

```text
cluster_split_v3_A2/sensitivity_analysis/4p5A/06_graph/train_pyg_3d_graph/
```

These samples are provided to illustrate data format and directory organization. They are not the full benchmark dataset used in the study.

---

## Requirements

The code uses Python together with scientific-computing, structural-bioinformatics, protein-language-model, and graph-learning packages.

Core dependencies include:

- Python
- PyTorch
- PyTorch Geometric
- NumPy
- pandas
- scikit-learn
- Biopython
- ESM-2 / protein language model dependencies

External preprocessing tools used by the workflow include:

- **CD-HIT**
- **DSSP / mkdssp**

For exact package versions, export the environment used for the experiments, for example:

```bash
pip freeze > requirements.txt
```

or, if Conda was used:

```bash
conda env export > environment.yml
```

---

## Running the pipeline

The scripts were developed as a research workflow and retain the project-specific path settings used during the experiments. Users running the repository on another machine should review the input/output paths in the relevant scripts before execution.

A typical execution order is:

```bash
python 00_filter_and_copy_pdb.py
python 00_trim_extra_chains.py
python 01_extract_complex_sequences.py
python 02_build_complex_cluster_mapping.py
python 02_complex_chain_processing_QC.py
python 03_cluster_level_split1.py
python 04_generate_labels_v3.py
python 05_build_graph.py
python 08_train.py
python 09_eval_best_model.py
```

Additional ablation, sensitivity-analysis, internal-comparison, and CR1 scripts can be run independently as required.

---

## Model weights

The trained model checkpoint(s) and model-evaluation outputs are stored in:

```text
cluster_split_v3_A2/partner_only_DSSP/07_model_save/
```

Before committing model weights, check the size of each `.pt`/`.pth` file.

If a checkpoint is too large for normal GitHub Git storage, use **Git LFS**:

```bash
git lfs install
git lfs track "cluster_split_v3_A2/partner_only_DSSP/07_model_save/*.pt"
git lfs track "cluster_split_v3_A2/partner_only_DSSP/07_model_save/*.pth"
git add .gitattributes
```

---

## Notes

- `.idea/` and `__pycache__/` are local development artifacts and should not be committed.
- The complete PDBbind dataset and full generated graph dataset are not included because of repository size and third-party redistribution considerations.
- `dssp.exe` is a third-party executable. Redistribute it only if its license permits; otherwise remove it from the public repository and instruct users to install DSSP separately.
- Empty directories are not tracked by Git. If you want an otherwise empty directory to remain visible on GitHub, place a small `.gitkeep` or `README.md` inside it.

---

## Citation

If you use EvoStruct-GAT, please cite the associated manuscript.

```text
Citation information will be updated after publication.
```

---

## License

Add an appropriate open-source license before public release if required by the authors/institution.

---

## Contact

For questions about the code or preprocessing workflow, please open an issue in this repository or contact the corresponding author.
