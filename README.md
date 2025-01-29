<div align="center">

  # 2SSP <br /> A Two-Stage Framework for Structured Pruning of LLMs

</div>

> **Abstract.** 
*We propose a novel Two-Stage framework for Structured Pruning (2SSP) for pruning Large Language Models (LLMs), which combines two different strategies of pruning, namely Width and Depth Pruning. The first stage (Width Pruning) removes entire neurons, hence their corresponding rows and columns, aiming to preserve the connectivity among the pruned structures in the intermediate state of the Feed-Forward Networks in each Transformer block. This is done based on an importance score measuring the impact of each neuron over the output magnitude. The second stage (Depth Pruning), instead, removes entire Attention submodules. This is done by applying an iterative process that removes the Attention submodules with the minimum impact on a given metric of interest (in our case, perplexity). We also propose a novel mechanism to balance the sparsity rate of the two stages w.r.t. to the desired global sparsity. We test 2SSP on four LLM families and three sparsity rates (25%, 37.5%, and 50%), measuring the resulting perplexity over three language modeling datasets as well as the performance over six downstream tasks. Our method consistently outperforms five state-of-the-art competitors over three language modeling and six downstream tasks, with an up to two-order-of-magnitude gain in terms of pruning time.*

## Installation

We recommend setting up the environment using either a Conda environment or a Python virtual environment. Our experiments were conducted using Python 3.11.7.

Clone the `2SSP` repository and install its dependencies:
```bash
git clone https://github.com/FabrizioSandri/2SSP.git
cd 2SSP
pip install -r requirements.txt
```

Install Language Model Evaluation Harness (required for downstream task evaluation):
```bash
cd lm_eval
pip install -e ./
```

## Usage

```
usage: main.py [-h] --model MODEL [--seed SEED] [--cache_dir CACHE_DIR] [--dense]
               [--pruning_method {2ssp,window_based,shortgpt,blockpruner,blockpruner_submodule,evopress,evopress_submodule,slicegpt}]
               [--num_prune NUM_PRUNE] [--ablation {calibration,one_stage,rows_vs_cols,l1_norm,balancing_sparsity_ratio}]
               [--main_table_results] [--evaluate_inference] [--evaluate_downstream] [--evaluate_perplexity]
               [--evaluate_qualitative] [--local_datasets]

Pruning of transformer models

options:
  -h, --help            show this help message and exit
  --model MODEL         Specify the model's name or path to be pruned
  --seed SEED           Set a seed for reproducibility (default: 0)
  --cache_dir CACHE_DIR
                        Path to a directory in which a downloaded pretrained model should be cached. This option is not supported
                        when --pruning_method=slicegpt
  --dense               Load the original dense model without pruning
  --pruning_method {2ssp,window_based,shortgpt,blockpruner,blockpruner_submodule,evopress,evopress_submodule,slicegpt}
                        Specify the pruning method to apply
  --num_prune NUM_PRUNE
                        Number of transformer blocks to prune. Use -1 to prune all possible blocks, -2 for specific sparsity ratios
                        (25%, 37.5%, 50%), or specify an integer value for exact number of blocks to prune
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