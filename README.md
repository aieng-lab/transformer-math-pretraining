> Jonathan Drechsel, Katja Noack, Anja Reusch, Steffen Herbold

[![arXiv](https://img.shields.io/badge/arXiv-2502.20855-B31B1B.svg)](https://arxiv.org/abs/2502.20855)

# Transformer Math Pretraining

Framework to pretrain mathematical aware transformer models, first introduced by [MAMUT: A Novel Framework for Modifying Mathematical Formulas for the Generation of Specialized Datasets for Language Model Training](https://github.com/aieng-lab/math-mutator).

## Installation

### 1 . Clone the repository:
```bash
git clone https://github.com/aieng-lab/transformer-math-pretraining.git
```

### 2. Create a Python Environment

#### Option 1: Using Conda
```bash
conda env create -f environment.yml
conda activate transformer_pretraining
```

#### Option 2: Using pip
```bash
python3 -m venv env_tp
source env_tp/bin/activate
pip install -r requirements.txt
```

### 3. Download the datasets
```bash
python src/execution/data/download_data.py
```


### 4. [Optional] Create a mathematical tokenizer
```bash
python src/execution/create_tokenizer.py
```
This script will first analyze the most common math tokens contained in MT, and then create a mathematical tokenizer for bert-base-cased with 300 additional math tokens (the most frequent 300 math tokens not found in the original tokenizer). Adjust the script to use, e.g., a different base model.

This does not only create the mathematical tokenizer, but also saves a model with randomly initialized weight for these added tokens (models/tokenized). 

## Pretraining Models

This repository can be used to pretrain mathematical aware transformer models based on the MAMUT-enhanced datasets: [Mathematical Formulas (MF)](https://huggingface.co/datasets/ddrg/math_formulas), [Mathematical Texts (MT)](https://huggingface.co/datasets/ddrg/math_text), [Named Math Formulas (NMF)](https://huggingface.co/datasets/ddrg/named_math_formulas), and [Math Formula Retrieval (MFR)](https://huggingface.co/datasets/ddrg/math_formula_retrieval).
To recreate one of the pretrained models in the MAMUT paper, you can refer to the scripts (e.g., [`scripts/BERT_MF_MT.sh`](scripts/Bert_MF_MT.sh)). To pretrain all models from the MAMUT paper, you can use [`scripts/mamut.sh`](scripts/mamut.sh).
Please note that this pretraining should be run on a machine with 8 A100 GPUs with at least 40GB of GPU memory each. The default pretraining takes about 12 hours per task.

> **Note:** You need to adjust the `base_dir` in each script!

### Pretraining Details
The [`src/execution/training/execute_pretraining`](src/execution/training/execute_pretraining.py) script is designed to automate the pretraining of transformer-based models (e.g., BERT) on custom objectives and datasets. It supports both single and multi-objective pretraining (either one-by-one or in parallel/mixed) and offers full control over training parameters via command-line arguments.

The script handles: 

- **Loading datasets:** Expects a local DatasetDict.
- **Dataset preprocessing:** Adjusts dataset to fit batch sizes and objective-specific needs (e.g., preparing the epoch-wise changing false examples for NMF and MFR)
- **Objective Management:** Supports primarily the four objectives used for MAMUT (MF, MT, NMF, MFR), however, other objectives are supported as well (MLM, NSP, SOP, and more), but you will need to provide then more information like files (this documentation focuses on the MATH objectives)
- **Training:** Runs training using an `Executor` that wraps around Hugging Face Transformers and accelerates training across devices.
- **Saving:** After training, the final model and tokenizer are saved.

#### Parameters (via CLI)

Parameter | Description
---------|------------
`--pretraining_obj` | Pretraining objective(s) to be used. Can be a single objective or a list of objectives (e.g., `MF_MT`). If `one_by_one` is `True`, the order of the list matters.
`--base_bert` | Hugging Face identifier or local path of input model
`--one_by_one` | In case of multiple pretraining tasks, whether to train them ony by one (i.e., finish the first task completly before training the 2nd one), or in a mixed way (changing the task after each batch, if multiple GPUs are used, multiple tasks are used for each optimization step, e.g., one task on four GPUs and the 2nd task on other four GPUs)
`--opt_steps` | Number of optimization steps for pretraining
`--interval_len` | Number of optimization steps between evaluation
`--batch` | Batch size for pretraining per GPU
`--lr` | Learning rate for pretraining
`--warmup` | Number of warmup steps for learning rate scheduler
`--num_gpus` | Number of GPUs used for pretraining


## Some Implementation Details

- Specific implementation of the mathematical objectives can be found in `src/pretraining_methods/mlm_like/MLM/prepare.py` and `src/pretraining_methods/nsp_like/NSP/prepare.py`
  - The mathematical words used for whole word masking for MT can be found in `src/pretraining_methods/mlm_like/MLM/prepare#math_words`
- The `src/data_sets/PreTrainingDataset.py` contains some advanced logic to realize the mixed multi-objective training and changing false examples for NMF and MFR every epoch. To support multi-GPUs, an advanced deterministically randomized mapping is applied only based on the index provided.
- The pretraining objectives have different names in the code than in the paper:
  - `MF` = `MFM` = `MLM_MATH`
  - `MT` = `MTM` = `MLM_MATH_TEXT`
  - `NMF` = `NFIR` (Named-Formula-IR)
  - `MFR` = `FFIR` (Formula-Formula-IR)


## CITATION
If you use this evaluation framework, please cite the following paper:
```bibtex
@misc{drechsel2025mamutnovelframeworkmodifying,
      title={{MAMUT}: A Novel Framework for Modifying Mathematical Formulas for the Generation of Specialized Datasets for Language Model Training}, 
      author={Jonathan Drechsel and Anja Reusch and Steffen Herbold},
      year={2025},
      eprint={2502.20855},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.20855}, 
}
```

## Authors
- Katja Noack (original implementation, [@katja98](https://github.com/katja98))
- Jonathan Drechsel (math adaption, [@jdrechsel13](https://github.com/jdrechsel13))