# pharmacogenomic-knowledge-graph

A concise snapshot describing the pharmacogenomic knowledge graph (PGx-KG) artifacts, dataset provenance, and the associated preprint describing the work. This repository stores the produced results and supporting artifacts used for analysis and dissemination.

## Summary

This project integrates pharmacogenomic entities and relations into a knowledge graph to support relation prediction and exploratory analyses. The artifacts in this repository include saved model checkpoints, ranked candidate lists (top-k), degree-based feature tensors, and visualization outputs that demonstrate key findings.

Key points:

- Focus: pharmacogenomic relation prediction and ego-graph analysis.
- Artifacts: pretrained model checkpoints, top-k candidate CSVs, degree/features tensors, and visualization images.
- Publication: corresponding preprint describing the methods and results is available (see Citation section).

## Dataset

The primary dataset used to construct and analyze the PGx-KG is available from Zenodo:

- Zenodo DOI: [10.5281/zenodo.17189995](https://doi.org/10.5281/zenodo.17189995)

Please consult the Zenodo landing page for license, metadata, and download instructions. Download the dataset files (for example nodes/edges tables) and place them in a `data/` directory at the repository root to reproduce data-processing steps locally.

## Repository contents

This repository snapshot focuses on the produced outputs and results. Notable directories and their purpose:

- `results/`
  - `models/` — Saved model checkpoint files (pretrained weights / saved checkpoints), e.g. `DRUG_CAUSES_ADR.pt`, `DRUG_IN_PATH.pt`, `GENE_AFFECTS_DRUG.pt`, `GENE_IN_PATH.pt`, `VAR_ASSOC_DIS.pt`.
  - `topk/` — CSV files with ranked candidate lists per relation (e.g., `DRUG_CAUSES_ADR_top200_only.csv`). Columns typically include subject, object, score, and rank.
  - `degree_features_train/` — Degree-based feature tensors used during training and evaluation (files ending in `.pt`).
  - `ego_graphs/` — PNG visualizations of ego-graphs for selected top predictions.

The repository contains the artifacts needed to reproduce evaluation and visualization steps; complete processing scripts and experimental notebooks used during development are not included in this snapshot.

## Environment and dependencies

Development and evaluation were performed using Python 3.8+ with a standard scientific stack. Key packages include:

- numpy
- pandas
- networkx
- matplotlib
- seaborn
- scikit-learn
- torch (PyTorch)
- tqdm
- (optional) torch-geometric for graph neural network experiments

To set up a minimal environment (example):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas networkx matplotlib seaborn scikit-learn torch tqdm
# Install torch-geometric separately following compatibility instructions if GNN experiments are needed
```

A `requirements.txt` file can be added to pin exact versions for reproducibility.

## Reproducing results

High-level steps to reproduce evaluation and visualization results locally:

1. Download the Zenodo dataset and place files under `data/`.
2. Install dependencies into a Python environment as shown above.
3. Use the saved artifacts in `results/` (models, degree features) to run evaluation scripts (not included) or to inspect the included CSVs and visualizations.

If you need the original data-processing or training scripts, please contact the authors or check other project branches where development artifacts may be available.

## Citation

If you use resources from this repository in your research, please cite the dataset and the work as follows:

- Dataset (preferred): Faruk, M.O. (2025). Pharmacogenomic Knowledge Graph (PGx-KG): Processed Dataset for Link Prediction. Zenodo. [https://doi.org/10.5281/zenodo.17189995](https://doi.org/10.5281/zenodo.17189995)
- Preprint (bioRxiv): [10.1101/2025.09.24.25336269](https://doi.org/10.1101/2025.09.24.25336269)

Suggested citation (adapt to your style):

Faruk, M.O. (2025). A large-scale pharmacogenomic knowledge graph for drug-gene-variant-disease discovery. bioRxiv. [https://doi.org/10.1101/2025.09.24.25336269](https://doi.org/10.1101/2025.09.24.25336269)

Repository:

Faerque. pharmacogenomic-knowledge-graph. GitHub: [https://github.com/Faerque/pharmacogenomic-knowledge-graph](https://github.com/Faerque/pharmacogenomic-knowledge-graph)

Recommended formatted references

APA style

- Dataset: Faruk, M.O. (2025). Pharmacogenomic Knowledge Graph (PGx-KG): Processed Dataset for Link Prediction. Zenodo. [https://doi.org/10.5281/zenodo.17189995](https://doi.org/10.5281/zenodo.17189995)
- Preprint: Faruk, M.O. (2025). A large-scale pharmacogenomic knowledge graph for drug-gene-variant-disease discovery. bioRxiv. [https://doi.org/10.1101/2025.09.24.25336269](https://doi.org/10.1101/2025.09.24.25336269)

IEEE style

- Dataset: M. O. Faruk, "Pharmacogenomic Knowledge Graph (PGx-KG): Processed Dataset for Link Prediction," Zenodo, 2025. DOI: 10.5281/zenodo.17189995.
- Preprint: M. O. Faruk, "A large-scale pharmacogenomic knowledge graph for drug-gene-variant-disease discovery," bioRxiv, 2025. DOI: 10.1101/2025.09.24.25336269.

## Acknowledgements

This README provides a static snapshot of the artifacts generated during this study. Users are advised to consult the Zenodo landing page to confirm dataset licensing and attribution information prior to reuse. Requests for additional materials, clarifications on reproducibility, or correspondence regarding the study should be directed to the corresponding author.

---

Last updated: 2025-09-26
