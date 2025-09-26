# pharmacogenomic-knowledge-graph

This repository contains the code, notebooks, intermediate outputs, trained models, and visualizations produced during the construction and analysis of a pharmacogenomic knowledge graph (PGx-KG). The work integrates pharmacogenomic entities and relations and demonstrates a workflow for dataset preprocessing, model training for relation tasks, and generation of top-k ego-graph visualizations.

Repository name: `pharmacogenomic-knowledge-graph`

---

## Overview

This project builds and analyzes a knowledge graph in the pharmacogenomics domain. The main contributions contained in this repository are:

- Data processing notebooks that prepare the PGx knowledge graph inputs.
- Training scripts and notebooks for relation prediction / graph tasks (trained model weights are included for reproducibility and demonstration).
- Generation of top-k candidate lists and ego-graph visualizations for selected relations.
- Saved model checkpoints and degree-based feature tensors used in downstream analyses.

The included artifacts are intended to allow replication of the reported analyses and to serve as a starting point for further research on pharmacogenomic knowledge graphs.

## Primary dataset

The full dataset used in this work is available from Zenodo at DOI: 10.5281/zenodo.17189995.

Direct DOI link: [https://doi.org/10.5281/zenodo.17189995](https://doi.org/10.5281/zenodo.17189995)

Please consult the Zenodo record for the dataset license, metadata, and citation information. Download the dataset and place the extracted files into a `data/` directory at the repository root before running the notebooks.

Example download using the DOI resolver (run locally):

```bash
# open the DOI landing page in a browser to download the dataset files
xdg-open "https://doi.org/10.5281/zenodo.17189995"
```

Download the individual files (e.g., nodes.csv, edges.zip, etc.) from the Zenodo page linked above.

## Repository structure (high-level)

The repository contains the following notable files and folders (these are present in the saved workspace snapshot):

- `kg-pgx-training (2).ipynb` — Notebook containing model training experiments and related analyses.
- `processsing_data_v2.ipynb` — Data processing and graph construction notebook (prepares inputs for training and evaluation).
- `Topk_ego_graph.ipynb` — Notebook used to generate top-k candidate lists and ego-graph visualizations for selected relations.
- `results/` — Directory with produced artifacts and outputs:
  - `models/` — Trained model checkpoint files (example names: `DRUG_CAUSES_ADR.pt`, `DRUG_IN_PATH.pt`, `GENE_AFFECTS_DRUG.pt`, `GENE_IN_PATH.pt`, `VAR_ASSOC_DIS.pt`).
  - `topk/` — CSV files with the top-k candidate lists per relation (e.g., `DRUG_CAUSES_ADR_top200_only.csv`).
  - `degree_features_train/` — Degree-based feature tensors used during training (files ending in `.pt`).
  - `ego_graphs/` — PNG visualizations of ego-graphs for selected top predictions.

Note: File names and locations reflect the current workspace snapshot and may be extended if additional experiments or outputs are produced.

## Environment and Dependencies

This project was developed using Python 3.8 or newer, with the following key packages:

- numpy
- pandas
- networkx
- matplotlib
- seaborn
- scikit-learn
- torch (PyTorch)
- tqdm

For graph neural network tasks, torch-geometric (PyG) and its dependencies were utilized, including CUDA-enabled builds for GPU acceleration.

A virtual environment was created and packages were installed via pip, for example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas networkx matplotlib seaborn scikit-learn torch tqdm
# torch-geometric was installed following the official instructions for the CUDA/PyTorch version
```

A `requirements.txt` file could be generated to lock exact package versions for reproducibility.

## Workflow Demonstrated

The project demonstrates a complete workflow for pharmacogenomic knowledge graph analysis:

1. The dataset was downloaded from Zenodo (DOI: 10.5281/zenodo.17189995) and extracted into a `data/` directory at the repository root.
2. A Python virtual environment was set up with the required dependencies as described above.
3. The notebooks were executed in sequence:

   - `processsing_data_v2.ipynb` — Processed the data to build the graph and generate training inputs, writing prepared files and tensors.
   - `kg-pgx-training (2).ipynb` — Conducted training and evaluation, saving model weights under `results/models/`.
   - `Topk_ego_graph.ipynb` — Generated top-k results and ego-graph visualizations, outputting images and CSVs to `results/ego_graphs/` and `results/topk/` respectively.

Notebooks were executed interactively using JupyterLab, with commands such as:

```bash
pip install jupyterlab
jupyter lab
```

For non-interactive execution, nbconvert was used, for example:

```bash
jupyter nbconvert --to notebook --execute "processsing_data_v2.ipynb" --output executed_processing.ipynb
```

## What is in `results/` (interpretation)

- `results/models/*.pt` — Saved PyTorch model checkpoints for relation prediction / graph tasks. These are provided for demonstration and quick evaluation; re-training can be demonstrated by re-executing the training notebook.
- `results/topk/*.csv` — Ranked candidates (top-k) produced for each evaluated relation. Each CSV lists subject, object, score, and rank (column names may vary by notebook implementation).
- `results/ego_graphs/*.png` — Ego-graph visualizations for top predictions; useful for inspection and qualitative analysis.
- `results/degree_features_train/*.pt` — Precomputed node-degree based features used as input to models (example file names: `ADR_deg3.pt`, `DRUG_deg3.pt`, etc.).

## Potential Extensions

The project could be extended by:

- Generating a `requirements.txt` or `environment.yml` file with pinned versions for exact reproducibility.
- Adding a `LICENSE` file indicating the intended license for the code in this repository (for example, MIT), while respecting the dataset license from Zenodo.
- Providing a supplementary document describing the experimental setup, hyperparameters, evaluation metrics, and interpretation of the generated top-k lists and visualizations.

## Citation

If you use the dataset in this repository in your research, please cite the Zenodo dataset and this repository. The Zenodo dataset DOI is:

10.5281/zenodo.17189995

Suggested citation format for the dataset (adapt to your citation style):

Author(s). Title. Zenodo. DOI:10.5281/zenodo.17189995

Also cite the repository if appropriate:

Faerque. pharmacogenomic-knowledge-graph. GitHub repository: [https://github.com/Faerque/pharmacogenomic-knowledge-graph](https://github.com/Faerque/pharmacogenomic-knowledge-graph)

Preprint (bioRxiv): https://doi.org/10.1101/2025.09.24.25336269

## Acknowledgements and notes

- This README is based on the repository snapshot and the outputs saved in `results/` observed in the workspace. If additional scripts or helper functions exist elsewhere in the project, include them in the repository root or a `src/` directory and update this README accordingly.
- Verify dataset license and attribution information on the Zenodo landing page before reuse.

---

Last updated: 2025-09-24
