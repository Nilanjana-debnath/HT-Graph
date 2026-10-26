# HT-Graph: Heterogeneous Continuous-Time Dynamic Graph Representation Learning

This repository contains the implementation of **HT-Graph**, a framework for temporal link prediction on heterogeneous continuous-time dynamic graphs (HCTDGs). HT-Graph combines heterogeneous event encoding, a neighbor-aware memory module, a bounded temporal Neighbor-Store, and a neighbor-aware restarter to support efficient temporal representation learning and multi-GPU training.

- **Paper:** [HT-Graph: Heterogeneous Continuous-Time Dynamic Graph Representation Learning using Neighbor-store with Restart](https://www.semanticscholar.org/paper/HT-Graph%3A-Heterogeneous-Continuous-Time-Dynamic-Debnath-U./f119c6faf851bb775d4ceb79c9c55107d814fd4b)
- **Paper PDF:** [Download / read the paper](https://unnikrishnan-c.github.io/papers/HTGraph.pdf)
- **Code:** [Nilanjana-debnath/HT-Graph](https://github.com/Nilanjana-debnath/HT-Graph)

> **Important:** The code repository does not bundle the datasets. Download the raw or prepared dataset files separately and place them in the directory structure expected by the scripts. The data and processed-data directories are excluded by `.gitignore`.

## Table of Contents

- [Overview](#overview)
- [Main Components](#main-components)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Datasets and Download Links](#datasets-and-download-links)
- [Data Preparation and Processing](#data-preparation-and-processing)
- [Training](#training)
- [Multi-GPU Training](#multi-gpu-training)
- [Outputs](#outputs)
- [Troubleshooting and Repository Notes](#troubleshooting-and-repository-notes)
- [Citation](#citation)

## Overview

Real-world interaction networks evolve over time. Nodes and interactions can have different types, and future interactions depend on both the history of events and the local graph structure. HT-Graph models this information in a continuous-time setting and uses a restart mechanism to make training on temporal chunks more parallelizable.

The paper evaluates HT-Graph on homogeneous and heterogeneous temporal graph datasets. The evaluation reports Average Precision (AP) and Area Under the ROC Curve (AUC), and also compares training efficiency and multi-GPU scalability. See the [paper](https://unnikrishnan-c.github.io/papers/HTGraph.pdf) for the mathematical details, experimental protocol, comparisons, and results.

## Main Components

### 1. Heterogeneous Event Encoding (HEE)

Events are represented using interaction/relation types and temporal information. The paper describes a preprocessing transformation that simplifies the graph's node-type representation while preserving relation-type information in the event stream.

### 2. Neighbor-aware memory

The memory module maintains evolving node states as historical interactions arrive. Messages are generated from event information and node states, aggregated, and used to update memory.

### 3. Neighbor-Store

The Neighbor-Store keeps a bounded temporal neighborhood for each node. The stored information includes neighbor identities, relation/event types, and timestamps. The bounded store reduces the need to repeatedly sample long histories.

### 4. Neighbor-aware restarter

The restarter estimates node memory states for intermediate temporal chunks. Together with a distillation loss that aligns restart estimates with the encoder's memory, it allows chunks of temporal events to be trained in parallel.

### 5. Link prediction

The decoder uses temporal node representations to score observed and negative interactions. Training is self-supervised: positive interactions and sampled negative links contribute to the link-prediction objective.

## Repository Layout

The current repository root contains these key implementation files and directories:

```text
HT-Graph/
├── tiger/                            # Model, data loader, graph and utility code
├── CHANGELOG.py                      # Model/version information
├── GAT.py                            # Attention-related implementation
├── data_preprocess.py                # Dataset-specific preprocessing helpers
├── init_utils.py                     # Argument parsing, data initialization and model setup
├── log.py                            # Logging utilities
├── parse_args.py                     # Additional/legacy argument parser
├── train_self_supervised.py          # Single-process training/evaluation entry point
├── train_self_supervised_newddp.py   # DDP / multi-GPU training entry point
├── train_self_supervisedDDP.py       # Additional DDP training variant
├── train_self_supervised_ddp.py      # Additional DDP training variant
├── train_self_supervised_ddp copy.py # Additional training variant
├── train_utils.py                    # Training helpers
├── run.sh                            # Example multi-GPU launch command
└── tests.sh                          # Dataset training commands
```

The `data/`, `processed/`, result, log, checkpoint, and saved-model directories are local runtime assets and are not included in the repository. The code imports `load_jodie_data` from `tiger.data.data_loader`; keep the `tiger/` package in place when running training.

## Requirements

The paper reports an experimental environment using:

- Python 3.10
- PyTorch 2.1.0
- CUDA 12.0
- Two NVIDIA GeForce RTX 2080 Ti GPUs for the multi-GPU experiments

The environment file shared for this repository contains the following dependencies:

- `pytorch`
- `torch-scatter`
- `pandas`
- `scikit-learn`
- `tqdm`
- `numpy`

The installed PyTorch, CUDA runtime, NVIDIA driver, and `torch-scatter` build must be mutually compatible. The version numbers above describe the paper's reported setup; they are not a guarantee that the same CUDA build is appropriate for every machine.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Nilanjana-debnath/HT-Graph.git
cd HT-Graph
```

### 2. Create `config_env.yaml`

Create this file in the repository root using the environment definition shared for this project:

```yaml
name: F_env
channels:
  - defaults
dependencies:
  - python=3.10
  - pytorch
  - conda-forge::torch-scatter
  - pandas
  - scikit-learn
  - tqdm
  - numpy
```

### 3. Create and activate the environment

```bash
conda env create -f config_env.yaml
conda activate F_env
```

If the environment already exists, activate it instead:

```bash
conda activate F_env
```

If Conda cannot solve a compatible `torch-scatter` build, install a build matching the installed PyTorch/CUDA versions using the [official PyTorch Geometric installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### 4. Check Python and GPU availability

```bash
python --version
python -c "import torch, numpy, pandas, sklearn, tqdm; print('Imports successful'); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

For the two-GPU example below, make sure both GPU 0 and GPU 1 are visible to the process.

## Datasets and Download Links

**Datasets must be downloaded separately.** The `.gitignore` file excludes both `data/` and `processed/`, so these datasets are not versioned in this repository. Download instructions and file requirements differ by dataset; use the raw files that match the filenames read by `data_preprocess.py`, or obtain a prepared dataset and convert it to the format expected by this implementation.

### Source links

| Dataset / group | Download or reference | Notes for this repository |
|---|---|---|
| Wikipedia and Reddit | [JODIE dataset page](https://snap.stanford.edu/jodie/) and [JODIE repository](https://github.com/claws-lab/jodie) | The Wikipedia preprocessor in this repo expects `data/wikipedia/wikipedia.csv`. The Reddit conversion in this repo expects an intermediate file under `processed/reddit/`; see [processing details](#data-preparation-and-processing). |
| Enron, SocialEvolve, UCI | [NAT dataset acquisition instructions](https://github.com/Graph-COM/Neighborhood-Aware-Temporal-Network#instructions-on-acquiring-datasets) | The linked project provides prepared versions of these temporal datasets. They are not guaranteed to be drop-in compatible with this repository's file names; convert/check the layout before training. |
| AskUbuntu | [SNAP Ask Ubuntu temporal network](https://snap.stanford.edu/data/sx-askubuntu.html) | The code expects the three relation files listed below. |
| MathOverflow | [SNAP MathOverflow temporal network](https://snap.stanford.edu/data/sx-mathoverflow.html) | The code expects the three relation files listed below. |
| SuperUser | [SNAP Super User temporal network](https://snap.stanford.edu/data/sx-superuser.html) | The code expects the three relation files listed below. |
| StackOverflow | [SNAP Stack Overflow temporal network](https://snap.stanford.edu/data/sx-stackoverflow.html) | The paper evaluates this dataset, but the current `data_preprocess.py` does not define a `process_stackoverflow()` function. An already prepared dataset or an additional converter is needed. |
| MovieLens | [GroupLens MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) | The code expects `u1.base`, `u1.test`, `u.user`, and `u.item` under `data/movielens/`; these are available in the MovieLens 100K archive. |
| UCI / CollegeMsg | [SNAP CollegeMsg temporal network](https://snap.stanford.edu/data/CollegeMsg.html) | This is a public source for the UCI/CollegeMsg interaction network. The current `process_uci()` helper expects `processed/uci/events2.csv`, not `CollegeMsg.txt` directly. An upstream conversion is required first. |
| SocialEvolve variants | [NAT repository and dataset instructions](https://github.com/Graph-COM/Neighborhood-Aware-Temporal-Network#instructions-on-acquiring-datasets) | The experiment script refers to `socialevolve_1month` and `socialevolve_2weeks`; this repository's helper is named `process_socialevolve_1m()` and does not generate both variants automatically. |
| tgbl-coin | [Temporal Graph Benchmark dataset documentation](https://tgb.complexdatalab.com/docs/linkprop/) | The paper's evaluation table includes `tgbl-coin`, but the current `tests.sh` and `data_preprocess.py` do not provide a complete end-to-end download/preprocessing path for it. |

The paper reports the following 11 evaluation datasets: **Tgbl-wiki-v2, Reddit, Enron, SocialEvolve, UCI, tgbl-coin, AskUbuntu, MathOverflow, MovieLens, SuperUser, and StackOverflow**. The repository's `tests.sh` uses a different list/alias set (`wikipedia`, `math`, and others); do not assume `wikipedia` and `Tgbl-wiki-v2`, or `math` and `MathOverflow`, are identical without checking the prepared data and evaluation protocol.

### Raw file layout expected by the current preprocessing helpers

Create directories as needed, then place raw files using the exact paths below:

```text
data/
├── wikipedia/
│   └── wikipedia.csv
├── askubuntu/
│   ├── sx-askubuntu-a2q.txt
│   ├── sx-askubuntu-c2a.txt
│   └── sx-askubuntu-c2q.txt
├── mathoverflow/
│   ├── sx-mathoverflow-a2q.txt
│   ├── sx-mathoverflow-c2a.txt
│   └── sx-mathoverflow-c2q.txt
├── superuser/
│   ├── sx-superuser-a2q.txt
│   ├── sx-superuser-c2a.txt
│   └── sx-superuser-c2q.txt
└── movielens/
    ├── u1.base
    ├── u1.test
    ├── u.user
    └── u.item
```

The Stack Exchange files are the three separate temporal interaction streams: answer-to-question (`a2q`), comment-to-answer (`c2a`), and comment-to-question (`c2q`). The preprocessing code combines them and assigns a different event type to each stream. The SNAP downloads are compressed (`.gz`) on the source pages; extract them and rename/place the resulting text files to match the paths above.

For datasets such as Enron, Reddit, SocialEvolve, SocialEvolve variants, and UCI, the current preprocessing helper reads an intermediate `events2.csv` from a directory under `processed/`. The upstream step that creates those `events2.csv` files is **not implemented as a complete downloader/preprocessor in this repository**, so download or generate those intermediates separately before running the corresponding conversion.

## Data Preparation and Processing

### What `data_preprocess.py` does

The preprocessing helper is dataset-specific rather than a general downloader. It reads files from fixed paths, normalizes event data, and writes files under `processed/<dataset>/`. The main transformations include:

1. **Read raw interactions or an intermediate event file.** Examples include the comma-separated Wikipedia CSV, the three Stack Exchange streams, the MovieLens 100K files, or an `events2.csv` prepared upstream.
2. **Build a chronological event stream.** Several converters sort events by timestamp and reset the row index before writing the normalized file.
3. **Map node IDs to contiguous indices.** The code reindexes nodes and reserves/pads index zero in relevant paths.
4. **Normalize temporal information.** The Wikipedia/Twitter paths subtract the minimum timestamp to start the timeline at zero.
5. **Encode event and node types.** AskUbuntu, MathOverflow, and SuperUser use event types 1, 2, and 3 for the three interaction streams. MovieLens represents users and movies as different node types and ratings as edge types.
6. **Create event IDs and metadata.** Relevant converters create sequential edge/event indices and write `desc.json` with counts such as number of nodes, events, node types, and edge types.
7. **Save features where implemented.** For example, the Wikipedia path writes `node_ft.npy` and `edge_ft.npy`; MovieLens writes a node feature array. The Stack Exchange converters do not save these feature arrays in their current functions, so verify that the model data loader handles the absent feature files as intended.

### Output artifacts

The exact outputs depend on the dataset-specific converter. The code writes to paths such as:

```text
processed/
├── wikipedia/
│   ├── events.csv
│   ├── node_ft.npy
│   ├── edge_ft.npy
│   └── desc.json
├── askubuntu/
│   ├── events.csv
│   └── desc.json
├── mathoverflow/
│   ├── events.csv
│   └── desc.json
├── superuser/
│   ├── events.csv
│   └── desc.json
└── movielens/
    ├── events.csv
    ├── events_test.csv
    ├── node_ft.npy
    └── desc.json
```

This tree is illustrative and shows the outputs that the corresponding converter writes; it is **not** a guarantee that every dataset has the same files. Some dataset paths also require additional intermediate files or feature arrays. The training loader is `load_jodie_data` imported from `tiger.data.data_loader`, so check that the selected dataset folder contains the files expected by that loader before starting training.

### How to run the converters safely

**Important implementation detail:** `data_preprocess.py` has no command-line `--data` selection. At the bottom of the current file, the active calls are:

```python
process('socialevolve')
process('socialevolve_1m')
process('uci')
```

These converters read `processed/socialevolve/events2.csv`, `processed/socialevolve_1m/events2.csv`, and `processed/uci/events2.csv`. Running `python data_preprocess.py` unchanged will attempt these conversions and can fail if the intermediate files are missing. The other `process(...)` calls at the bottom are commented out.

To process a raw dataset with a converter already implemented in the file:

1. Download and extract its raw files to the exact paths shown above.
2. Open `data_preprocess.py` and go to the final block.
3. Comment out the currently active calls unless the required `events2.csv` files already exist.
4. Uncomment only the converter call needed for the dataset, for example `process('wikipedia')` or `process('askubuntu')`.
5. Run the preprocessing script from the repository root:

```bash
python data_preprocess.py
```

6. Check that the corresponding `processed/<dataset>/` outputs were created before running training.

The currently implemented raw-data converters include `wikipedia`, `twitter`, `mathoverflow`, `askubuntu`, `superuser`, and `movielens`. The `enron`, `reddit`, `socialevolve`, `socialevolve_1m`, and `uci` functions convert an existing `events2.csv` into the normalized `events.csv` structure. The current helper does not define `process_math()` or `process_stackoverflow()`, and its `socialevolve_1m` function does not automatically produce both `socialevolve_1month` and `socialevolve_2weeks` variants.

### Training data split and `--subset`

The training code loads the dataset through `load_jodie_data`, then builds the temporal graph and training/evaluation data loaders. It also prepares validation/test data, including inductive evaluation data.

The `--subset` parameter is implemented in `init_utils.py`: for values below `1.0`, it keeps the first `ceil(len(train_data) * subset)` examples as the training subset and assigns the remainder of the original training partition to an offline data loader. Therefore, `--subset 0.2` means the first 20% of the existing training partition in its current order; it is **not** implemented as a random 20% sample.

## Training

Activate the Conda environment first:

```bash
conda activate F_env
```

### 1. Run one self-supervised experiment

For example, train on Wikipedia using 20% of the training partition:

```bash
python train_self_supervised.py --data wikipedia --subset 0.2
```

Use the actual dataset name and prepared files available in `processed/`. The script accepts `--data` and `--subset`; additional options can be inspected with:

```bash
python train_self_supervised.py --help
```

### 2. Run the dataset commands from `tests.sh`

The following preserves the distinct dataset commands in `tests.sh` while removing the duplicate AskUbuntu run:

```bash
python train_self_supervised.py --data wikipedia --subset 0.2
python train_self_supervised.py --data reddit --subset 0.2
python train_self_supervised.py --data enron --subset 0.2
python train_self_supervised.py --data math --subset 0.2
python train_self_supervised.py --data socialevolve --subset 0.2
python train_self_supervised.py --data askubuntu --subset 0.2
python train_self_supervised.py --data uci --subset 0.2
python train_self_supervised.py --data mathoverflow --subset 0.2
python train_self_supervised.py --data movielens --subset 0.2
python train_self_supervised.py --data superuser --subset 0.2
```

Optional commands present in `tests.sh` are:

```bash
python train_self_supervised.py --data socialevolve_1month --subset 0.2
python train_self_supervised.py --data socialevolve_2weeks --subset 0.2
```

Run these only after preparing folders and files for those exact dataset names. In particular, the current `data_preprocess.py` does not generate these two variants from raw data on its own. Likewise, `math` appears in `tests.sh` but there is no `process_math()` function in the current preprocessing helper; the corresponding processed dataset must already exist and match the loader's expected format.

### 3. Run all standard commands sequentially

Save the following as `run_all_experiments.sh`:

```bash
#!/usr/bin/env bash
set -e

# Activate F_env before running this script.
datasets=(
    wikipedia
    reddit
    enron
    math
    socialevolve
    askubuntu
    uci
    mathoverflow
    movielens
    superuser
)

for dataset in "${datasets[@]}"; do
    echo "========================================"
    echo "Starting HT-Graph on: ${dataset}"
    echo "========================================"

    python train_self_supervised.py \
        --data "${dataset}" \
        --subset 0.2

done

echo "All requested experiments completed."
```

Run it from the repository root:

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

The script runs experiments sequentially and stops on the first failing command because `set -e` is enabled. Make sure all listed dataset folders are prepared first; otherwise it will stop at the first missing or invalid dataset.

## Multi-GPU Training

The repository's `run.sh` launches `train_self_supervised_newddp.py` on GPUs 0 and 1:

```bash
python3 train_self_supervised_newddp.py \
    --gpu 0,1 \
    --prefix 241444 \
    --data askubuntu
```

Arguments:

- `--gpu 0,1`: selects GPU IDs 0 and 1. The script splits this comma-separated argument for distributed execution.
- `--prefix 241444`: supplies an experiment prefix used in output/checkpoint names.
- `--data askubuntu`: selects the dataset. The dataset must already be prepared in the expected layout.

To run the checked-in shell script:

```bash
chmod +x run.sh
./run.sh
```

To change the experiment, edit the `--gpu`, `--prefix`, or `--data` arguments in `run.sh`. Run only on GPUs visible to the current environment. For command-line options supported by this particular script, use:

```bash
python train_self_supervised_newddp.py --help
```

## Outputs

The training code creates output directories when needed. The single-process entry point writes model checkpoints and experiment results under paths including:

- `saved_models/` — saved model state.
- `saved_checkpoints/<experiment>/` — epoch checkpoints.
- `results/` — JSON result records and pickle files.

The actual filenames include the experiment prefix and a hash of the run configuration. The `.gitignore` excludes these generated directories; back them up separately if needed.


## Citation

If you use HT-Graph in your research, please cite the paper and verify the final publication metadata on the [Semantic Scholar paper page](https://www.semanticscholar.org/paper/HT-Graph%3A-Heterogeneous-Continuous-Time-Dynamic-Debnath-U./f119c6faf851bb775d4ceb79c9c55107d814fd4b).

```bibtex
@inproceedings{debnath2025htgraph,
  title     = {HT-Graph: Heterogeneous Continuous-Time Dynamic Graph Representation Learning using Neighbor-store with Restart},
  author    = {Debnath, Nilanjana and Unnikrishnan, C.},
  booktitle = {Proceedings of the International Joint Conference on Neural Networks (IJCNN)},
  year      = {2025}
}
```

## Acknowledgements

HT-Graph was developed as research work at the Indian Institute of Technology Palakkad. Please consult the paper for full methodology, experimental details, baseline comparisons, and acknowledgements.
