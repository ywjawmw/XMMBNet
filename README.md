# XMMBNet: Explainable Multi-view Enhanced Multi-scale Biological Network Fusion Model for Drug Repurposing.

XMMBNet constructs drug and disease similarity networks from 12 types of information spanning molecular, functional, and pathway levels and integrates six biological networks. It organizes these networks into direct therapeutic and indirect biological views, and uses network-specific graph encoders, dynamic entity aggregation, and multi-view contrastive learning to coordinate complementary evidence while preserving their distinct biological roles. XMMBNet further combines learned evidence weights with biological network subgraphs and protein-guided paths to generate traceable mechanistic hypotheses through constrained language generation and quality control.
## Repository structure

```text
XMMBNet/
├── drug_train_GCL_enhanced15.py              # Main 10-fold training entry point
├── drug_test_GCL_enhanced15_graph_info.py    # Prediction, subgraph and PPI-path export
├── dataset_loader_enhanced15.py               # Dataset loading and graph construction
├── model_proj_enhanced15.py                   # XMMBNet model
├── model_proj_enhanced15_test.py              # Test-time model components
├── evaluate15.py                              # Training evaluation
├── evaluate15_test.py                         # Test-time evaluation and explanations
├── GCL_layer.py                               # Graph contrastive-learning layers
├── utils*.py                                  # Graph and training utilities
├── requirement.txt                            # Python dependencies
├── name_data/drug_data/
│   ├── Adataset/
│   ├── Cdataset/
│   └── Gdataset/
├── explain/
│   ├── get_graph_info_ppi_path.py             # Build candidate-specific graph and PPI evidence JSON
│   ├── get_LLM_response.py                    # Generate the final LLM explanation
│   ├── Openai.py                              # LLM API wrapper
│   └── graph_information/                     # JSON evidence and LLM outputs
├── case/                                      # Case-study Excel outputs
├── result/                                    # Results
└── weight/                                    # Model checkpoints (created during training)
```

## Datasets

The three datasets are stored under `name_data/drug_data/` and are selected with `--data_name`.

| Dataset | Drugs | Diseases | Proteins |
|---|---:|---:|---:|
| Adataset | 1,220 | 2,480 | 3,710 |
| Cdataset | 663 | 409 | 927 |
| Gdataset | 593 | 313 | 2,670 |

The loader uses the following core files in each dataset directory:

- `drug_dis.csv`: drug–disease association matrix;
- `drug_sim.csv` and `dis_sim.csv`: drug and disease similarity matrices;
- `drug_pro_enhanced.csv` and `pro_dis_enhanced.csv`: drug–protein and protein–disease networks;
- `pro_sim_enhanced.csv`: protein similarity matrix;
- `ppi_adj_enhanced-{ppi_radio}.csv`: thresholded PPI adjacency matrix;
- `gene_emb.pt`: pretrained protein embeddings;
- `drug_name.csv`, `disease_name.csv`, and protein-name mapping files: entity identifiers and names.

The explanation scripts additionally use `drugbank_drugs.csv`, `omim_diseases.csv`, and `protein_name_enhanced.csv`. These files and the current case-study workflow are prepared for **Adataset**. Training supports all three datasets.

## Environment

The recommended environment uses Python 3.10, PyTorch 2.4.0, a CUDA-enabled DGL build, and an NVIDIA GPU.

```bash
conda create -n xmmbnet python=3.10 -y
conda activate xmmbnet
pip install -r requirement.txt
```

The LLM wrapper uses the `openai.ChatCompletion` interface provided by OpenAI Python releases earlier than 1.0:

```bash
pip install "openai<1"
```

Run all model commands from the repository root because the data and output paths in the scripts are relative paths.

## Training

The main training entry point is `drug_train_GCL_enhanced15.py`. It performs 10-fold cross-validation, evaluates AUROC and AUPR, and saves the best checkpoint for every fold.

Example for Adataset on GPU 0:

```bash
python drug_train_GCL_enhanced15.py \
  --data_name Adataset \
  --device 0 \
  --train_lr 0.03 \
  --layers 1 \
  --E_layers 1 \
  --dropout 0.25 \
  --num_neighbor 20 \
  --ppi_radio 0.4 \
  --tau 0.1 \
  --beta 0.1 \
  --lambda_margin 0.02 \
  --train_max_iter 5000 \
  --save_name Adataset_XMMBNet
```

The explicit model settings above are the best Adataset hyperparameters reported in Supplementary Table 1:

| Command-line argument | Value |
|---|---:|
| `--train_lr` | 0.03 |
| `--layers` | 1 |
| `--E_layers` | 1 |
| `--dropout` | 0.25 |
| `--num_neighbor` | 20 |
| `--ppi_radio` | 0.4 |
| `--tau` | 0.1 |
| `--beta` | 0.1 |
| `--lambda_margin` | 0.02 |

These values are also the current defaults in both the training and graph-information test scripts.

Training also supports `--data_name Cdataset` and `--data_name Gdataset` with their corresponding dataset-specific settings. Use `--device -1` for CPU execution.

The main outputs are:

- `weight/<dataset>_0time/*.pkl`: best checkpoint for each fold.

`--ppi_radio` selects a file named `ppi_adj_enhanced-<value>.csv`; only use a threshold for which that file exists in the selected dataset directory.

## Prediction and explanation workflow

The complete explanation workflow has three stages.

### 1. Export candidate predictions, subgraphs, and PPI paths

After training, run the graph-information test script:

```bash
python drug_test_GCL_enhanced15_graph_info.py \
  --data_name Adataset \
  --device 0 \
  --graph_info_disease_name MIM145500 \
  --graph_info_topn 10 \
  --graph_info_fold 2 \
  --protein_path_top_id 1
```

`--graph_info_disease_name` must exactly match a value in `name_data/drug_data/Adataset/disease_name.csv`. It takes precedence over `--graph_info_disease_id`. The script runs all ten checkpoints, ranks candidate drugs across folds, and exports:

- fold-wise and ensemble candidate rankings;
- drug-side and disease-side graph summaries;
- detailed contributing neighbors;
- direct protein overlap or PPI paths of up to three hops.

The Excel file is written under `case/Adataset/`. Set `OMIM_name` near the bottom of `drug_test_GCL_enhanced15_graph_info.py` to the short label used for the case-study output filename.

### 2. Build JSON evidence for a selected candidate

Set the case label, candidate rank, fold, and disease ID in `explain/get_graph_info_ppi_path.py`:

```python
FOLD_ID = 2
summary_rank_id = 1
Disease_NMAE = "FA"
# ...
disease_id = 439
```

`Disease_NMAE` and `summary_rank_id` select the Excel file exported in the previous stage, `FOLD_ID` selects its fold-specific worksheets, and `disease_id` selects the disease record. The script combines the selected drug and disease names, graph scores and neighbors, PPI status, and one- to three-hop PPI paths into a structured JSON file. Run it from the repository root:

```bash
python explain/get_graph_info_ppi_path.py
```

The result is written to:

```text
explain/graph_information/<case>_graph_info_ppi_path_top<rank>_fold<fold>.json
```

### 3. Generate the LLM explanation

Set the input `file_name`, `model_name`, and API configuration in `explain/get_LLM_response.py` and `explain/Openai.py`. The script uses paths relative to the `explain/` directory, so run it as follows:

```bash
cd explain
python get_LLM_response.py
```

The generated per-example responses and merged output are saved below `explain/graph_information/<model>_explain_DDA/`.

## Citation

If XMMBNet is useful in your research, please cite it as follows:

```bibtex
@misc{yue2026XMMBNet,
  title   = {XMMBNet: Explainable Multi-view Enhanced Multi-scale Biological Network Fusion Model for Drug Repurposing},
  author  = {Yue, Wenjing and Lu, Jinyuan and Gu, Wenjing and Chen, Hongyu and Liu, Anrong and Tian, Saisai and Zhang, Weidong},
  howpublished = {https://github.com/ywjawmw/XMMBNet},
  year    = {2026}
}
```
