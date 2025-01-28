# 2SSP: A Two-Stage Framework for Structured Pruning of LLMs

**Official PyTorch Implementation of 2SSP**

We introduce 2SSP, a novel Two-Stage Framework for Structured Pruning of
Large Language Models (LLMs). 2SSP synergistically combines Width
Pruning and Depth Pruning to achieve superior efficiency while maintaining
performance.

- Width Pruning (Stage 1): Eliminates entire neurons and their associated
  rows/columns to preserve connectivity in Feed-Forward Networks within
  Transformer blocks. Neuron importance is determined by their contribution to
  output magnitude.  
- Depth Pruning (Stage 2): Iteratively removes Attention submodules with minimal
  impact on a target metric (e.g., perplexity).


We evaluate 2SSP on four LLM families and test it across three sparsity rates
(25%, 37.5%, 50%) using three language modeling datasets and six downstream
tasks. 2SSP consistently outperforms five state-of-the-art methods while
achieving up to a two-order-of-magnitude reduction in pruning time.

## Installation

1. Clone `2ssp` and install its dependencies:
```bash
git clone https://github.com/FabrizioSandri/2SSP.git
cd 2SSP
pip install -e ./
```

2. Install Language Model Evaluation Harness (required for downstream task evaluation):
```bash
cd lm_eval
pip install -e ./
```

## Usage

```
usage: main.py [-h] --model MODEL [--seed SEED] [--cache_dir CACHE_DIR] [--dense]
               [--pruning_method {2ssp,window_based,shortgpt,blockpruner,blockpruner_submodule,evopress,evopress_submodule,slicegpt}]
               [--num_prune NUM_PRUNE]
               [--ablation {calibration,one_stage,rows_vs_cols,l1_norm,balancing_sparsity_ratio}]
               [--main_table_results] [--evaluate_inference] [--evaluate_downstream] [--evaluate_perplexity]
               [--evaluate_qualitative] [--local_datasets]

Pruning of transformer models

options:
  -h, --help            show this help message and exit
  --model MODEL         Specify the model's name or path to be pruned
  --seed SEED           Set a seed for reproducibility (default: 0)
  --cache_dir CACHE_DIR
                        Path to a directory in which a downloaded pretrained model feature extractor should be
                        cached
  --dense               Load the original dense model without pruning
  --pruning_method {2ssp,window_based,shortgpt,blockpruner,blockpruner_submodule,evopress,evopress_submodule,slicegpt}
                        Specify the pruning method to apply
  --num_prune NUM_PRUNE
                        Number of transformer blocks to prune. Use -1 to prune all possible blocks, -2 for
                        specific sparsity ratios (25%, 37.5%, 50%), or specify an integer value for exact number
                        of blocks to prune
  --ablation {calibration,one_stage,rows_vs_cols,l1_norm,balancing_sparsity_ratio}
                        Run a specific ablation experiment
  --main_table_results  Generate results for the main results table in the paper (Table 1)
  --evaluate_inference  Measure the model's inference time
  --evaluate_downstream
                        Perform downstream task evaluation at 37.5% sparsity
  --evaluate_perplexity
                        Evaluates perplexity on Wikitext2 only
  --evaluate_qualitative
                        Qualitative results
  --local_datasets      Use local datasets stored in the './data/' folder
```

#### Examples
- Dense Model Perplexity Evaluation:
   ```bash
   python 2ssp/main.py --model=meta-llama/Llama-2-7b-hf --dense --evaluate_perplexity
   ```

- Pruning at 50% sparsity(the equivalent of 16 blocks in Llama) with `2ssp` and evaluating perplexity:
   ```bash
   python 2ssp/main.py --model=meta-llama/Llama-2-7b-hf --pruning_method=2ssp --num_prune=16 --evaluate_perplexity
   ```
- Pruning at 50% sparsity(the equivalent of 16 blocks in Llama) with `ShortGPT` and evaluating perplexity:
   ```bash
   python 2ssp/main.py --model=meta-llama/Llama-2-7b-hf --pruning_method=shortgpt --num_prune=16 --evaluate_perplexity
   ```

- Generate main table results for 2SSP at 25%, 37.5% and 50% sparsity on Mistral:
   ```bash
   python 2ssp/main.py --model=mistralai/Mistral-7B-v0.3 --pruning_method=2ssp --num_prune=-2 --main_table_results
   ```

- Evaluate downstream tasks at 37.5% sparsity:
   ```bash
   python 2ssp/main.py --model=mistralai/Mistral-7B-v0.3 --pruning_method=2ssp --num_prune=12 --evaluate_downstream
   ```

## Supported Pruning Methods

- ShortGPT: [https://arxiv.org/abs/2403.03853](https://arxiv.org/abs/2403.03853)
- Window-Based: [https://arxiv.org/abs/2403.17887](https://arxiv.org/abs/2403.17887)
- SliceGPT: [https://arxiv.org/abs/2401.15024](https://arxiv.org/abs/2401.15024)
- BlockPruner: [https://arxiv.org/abs/2406.10594](https://arxiv.org/abs/2406.10594)
- EvoPress: [https://arxiv.org/abs/2410.14649](https://arxiv.org/abs/2410.14649)

---

## Acknowledgments

Our method includes code sourced from the following repositories:
- [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [EvoPress](https://github.com/IST-DASLab/EvoPress)
- [Transformer Compression with SliceGPT](https://github.com/microsoft/TransformerCompression/)

For more details, refer to the documentation or the associated research paper.