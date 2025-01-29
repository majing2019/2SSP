import argparse
import gc
import logging
import math
import time
import torch
from transformers import AutoTokenizer

from src.evaluation import *
from src.utilities import *
from src.datasets import *
from src.pruning import *
from src.evopress import *
from src.slicegpt import *
from src.ablations import *


def parse_args():
  parser = argparse.ArgumentParser(description="Pruning of transformer models")
  parser.add_argument('--model', type=str, required=True, help="Specify the model's name or path to be pruned")
  parser.add_argument('--seed', type=int, default=0, help="Set a seed for reproducibility (default: 0)")
  parser.add_argument('--cache_dir', type=str, required=False, help="Path to a directory in which a downloaded pretrained model should be cached. This option is not supported when --pruning_method=slicegpt")

  parser.add_argument(
    '--dense', 
    help="Load the original dense model without pruning", 
    action='store_true'
  )

  parser.add_argument(
    '--pruning_method', 
    type=str, 
    choices=['2ssp', 'window_based', 'shortgpt', 'blockpruner', 'blockpruner_submodule', 'evopress', 'evopress_submodule', 'slicegpt'],
    help="Specify the pruning method to apply"
  )
  parser.add_argument(
    '--num_prune', 
    type=int, 
    help="Number of transformer blocks to prune. Use -1 to prune all possible blocks, -2 for specific sparsity ratios (25%%, 37.5%%, 50%%), or specify an integer value for exact number of blocks to prune"
  )

  parser.add_argument(
    '--ablation', 
    type=str, 
    choices=['calibration', 'one_stage', 'rows_vs_cols', 'l1_norm', 'balancing_sparsity_ratio'], 
    required=False, 
    help="Run a specific ablation experiment"
  )

  parser.add_argument('--main_table_results', help="Generate results for the main results table in the paper (Table 1)", action='store_true')
  parser.add_argument('--evaluate_inference', help="Measure the model's inference time", action='store_true')
  parser.add_argument('--evaluate_downstream', help="Perform downstream task evaluation at 37.5%% sparsity", action='store_true')
  parser.add_argument('--evaluate_perplexity', help="Evaluates perplexity on Wikitext2 only", action='store_true')
  parser.add_argument('--evaluate_qualitative', help="Qualitative results", action='store_true')

  parser.add_argument('--local_datasets', help="Use local datasets stored in the './data/' folder", action='store_true')

  return parser.parse_args()


@torch.no_grad()
def main():
  logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
  )
  args = parse_args()
  set_seed(args.seed)
  
  # Load the tokenizer
  logging.info("Loading the tokenizer")
  tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
  logging.info("Loaded the tokenizer")

  ###################### Datasets 
  logging.info("Loading the Datasets")
    
  # Evaluation datasets
  dataset_wikitext = load_wikitext2(args.local_datasets)
  dataset_c4_val = load_c4(train=False, local=args.local_datasets)
  dataset_fineweb_edu = load_fineweb_edu(local=args.local_datasets)[:500]
  dataset_c4_train = load_c4(train=True, local=args.local_datasets)
  logging.info("Loaded the Datasets")

  logging.info("Tokenizing the Datasets")
  wikitext_input_ids = tokenizer("\n\n".join(dataset_wikitext["text"]), return_tensors="pt", add_special_tokens=False).input_ids
  c4_val_input_ids = tokenizer("\n\n".join(dataset_c4_val["text"]), return_tensors="pt", add_special_tokens=False).input_ids
  fineweb_edu_input_ids = tokenizer("\n\n".join(dataset_fineweb_edu["text"]), return_tensors="pt", add_special_tokens=False).input_ids
  logging.info("Tokenized the datasets")

  # Calibration datasets
  num_calibration_samples_2ssp = 32
  num_calibration_samples = 256 # For SliceGPT, ShortGPT and Window Based

  calibration_dataset = get_calibration(dataset_c4_train, tokenizer, num_samples=num_calibration_samples, seq_len=2048)

  calibration_dataset_2ssp = calibration_dataset[:num_calibration_samples_2ssp]
  first_calibration_sample = calibration_dataset[0]
  

  ###################### Dense model
  if args.dense:

    logging.info("Dense model evaluation")
    logging.info("Loading the model")    
    model = loadModel(args.model, args.cache_dir)
    logging.debug(model)
    printModelStats(model, "Dense model")
    
    if args.evaluate_inference == True:
      evaluate_inference_time(model, first_calibration_sample)

    if args.evaluate_downstream == True:
      evaluation_downstream(model, args.model)

    if args.main_table_results == True:
      evaluation_ppl(model, wikitext_input_ids, c4_val_input_ids, fineweb_edu_input_ids)
    
    if args.evaluate_perplexity == True:
      ppl = evaluate_perplexity(model, wikitext_input_ids, seq_len=2048)
      logging.info(f"Perplexity (wikitext2): {ppl}")  

    if args.evaluate_qualitative == True:
      qualitative_results(model, tokenizer, max_length=128)
    

  ###################### Pruning

  pruning_method = args.pruning_method
  if pruning_method is not None:

    num_prune = args.num_prune

    logging.info("Loading the model")
    model = loadModel(args.model, args.cache_dir)
    num_blocks = len(model.model.layers)
    logging.debug(model)

    if pruning_method == "slicegpt":
      del model # the model will be loaded by the SliceGPT model adapter
      gc.collect()
      torch.cuda.empty_cache()
      

    if num_prune == -1: # prune all possible blocks
      pruning_ratios = range(1,num_blocks - 1)
    elif num_prune == -2: # prune at 25%, 37.5%, 50%
      pruning_ratios = [0.25 * num_blocks, 0.375 * num_blocks, 0.5 * num_blocks]
      pruning_ratios = [int(math.ceil(i)) for i in pruning_ratios] # ceil for models like qwen where s * num_blocks is not an int
    else: # Prune a single sparsity rate
      pruning_ratios = range(num_prune, num_prune+1)

    for num_prune in pruning_ratios:
      logging.info(f"Pruning rate {num_prune / num_blocks} the equivalent of {num_prune} blocks")

      set_seed(args.seed)

      # Measure pruning time
      start_time = time.time()

      attnMask = None
      mlpMask = None
      if pruning_method == "window_based":
        mask = window_based(model, num_prune, calibration_dataset)
        attnMask = mask
        mlpMask = mask
      elif pruning_method == "shortgpt":
        mask = shortGPT(model, num_prune, calibration_dataset)
        attnMask = mask
        mlpMask = mask
      elif pruning_method == "blockpruner":
        mask = blockpruner(model, num_prune, first_calibration_sample)
        attnMask = mask
        mlpMask = mask
      elif pruning_method == "blockpruner_submodule":
        attnMask, mlpMask = blockpruner_submodule(model, num_prune, first_calibration_sample)
      elif pruning_method == "evopress":
        mask = evopress(model, num_prune, tokenizer, dataset_c4_train, drop_entire_block=True)
        attnMask = mask
        mlpMask = mask
      elif pruning_method == "evopress_submodule":
        attnMask, mlpMask = evopress(model, num_prune, tokenizer, dataset_c4_train, drop_entire_block=False)
      elif pruning_method == "2ssp":
        pruning_rate = num_prune / len(model.model.layers)
        model = two_stage_2ssp(model, calibration_dataset_2ssp, pruning_rate)
      elif pruning_method == "slicegpt":
        model = slicegpt(args.model, num_prune, calibration_dataset)
      else:
        logging.error("Invalid method provided")
        exit(1)

      end_time = time.time()
      logging.info(f"Pruning Time: {end_time - start_time} s")

      printModelStats(model, "Pruned model")
      
      # 2SSP: no masks are generated. The parameters are removed
      if attnMask is None:
        
        if args.evaluate_inference == True:
          evaluate_inference_time(model, first_calibration_sample)
        
        if args.evaluate_downstream == True:
          evaluation_downstream(model, args.model)

        if args.main_table_results == True:
          evaluation_ppl(model, wikitext_input_ids, c4_val_input_ids, fineweb_edu_input_ids)         
        
        if args.evaluate_perplexity == True:
          ppl = evaluate_perplexity(model, wikitext_input_ids, seq_len=2048)
          logging.info(f"Perplexity (wikitext2): {ppl}")       

        if args.evaluate_qualitative == True:
          qualitative_results(model, tokenizer, max_length=128)

        reset_mlps_shape(model)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if pruning_method != "slicegpt":
          model = loadModel(args.model, args.cache_dir)

      else:
        logging.debug(f"Pruned blocks: attn={attnMask} mlp={mlpMask}")

        maskModel(model, attnMask=attnMask, mlpMask=mlpMask)

        if args.evaluate_inference == True:
          evaluate_inference_time(model, first_calibration_sample)

        if args.evaluate_downstream == True:
          evaluation_downstream(model, args.model)

        if args.main_table_results == True:
          evaluation_ppl(model, wikitext_input_ids, c4_val_input_ids, fineweb_edu_input_ids)
        
        if args.evaluate_perplexity == True:
          ppl = evaluate_perplexity(model, wikitext_input_ids, seq_len=2048)
          logging.info(f"Perplexity (wikitext2): {ppl}")

        if args.evaluate_qualitative == True:
          qualitative_results(model, tokenizer, max_length=128)
        
        unmaskModel(model, attnMask=attnMask, mlpMask=mlpMask)

  ###################### Ablations Studies
  if args.ablation is not None:

    if args.ablation == "calibration":
      logging.info('Running ablation: Choice of Calibration Set Size')
      calibration_sizes = [2, 4, 8, 16, 32, 64, 128, 256]

      sparsity = 0.5
      ablation_calibration_dataset(args.model, tokenizer, sparsity, dataset_c4_train, wikitext_input_ids, calibration_sizes, seq_len=2048, method="2ssp", cache_dir=args.cache_dir)
    
    elif args.ablation == "one_stage":
      logging.info('Running ablation: Running stage 1 only')

      pruning_rates = [0.25, 0.375, 0.5]
      for pruning_rate in pruning_rates:
        model = loadModel(args.model, args.cache_dir)
        set_seed(args.seed)

        model = one_stage_2ssp(model, calibration_dataset_2ssp, pruning_rate)
        ppl = evaluate_perplexity(model, wikitext_input_ids, seq_len=2048)

        logging.info(f"Perplexity @ {pruning_rate} : {ppl}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    elif args.ablation == "rows_vs_cols":
      logging.info('Running ablation: Pruning Rows-Columns vs. Columns-Rows')

      pruning_rates = [0.25, 0.375, 0.5]
      for pruning_rate in pruning_rates:
        model = loadModel(args.model, args.cache_dir)
        set_seed(args.seed)

        model = two_stage_2ssp_inverted(model, calibration_dataset_2ssp, pruning_rate)
        ppl = evaluate_perplexity(model, wikitext_input_ids, seq_len=2048)

        logging.info(f"Perplexity @ {pruning_rate} : {ppl}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    elif args.ablation == "l1_norm":
      logging.info('Running ablation: Neuron Selection based on L1 norm')
      
      pruning_rates = [0.25, 0.375, 0.5]
      for pruning_rate in pruning_rates:
        model = loadModel(args.model, args.cache_dir)
        set_seed(args.seed)

        model = two_stage_2ssp_l1_norm(model, calibration_dataset_2ssp, pruning_rate)
        ppl = evaluate_perplexity(model, wikitext_input_ids, seq_len=2048)

        logging.info(f"Perplexity @ {pruning_rate} : {ppl}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    
    elif args.ablation == "balancing_sparsity_ratio":
      logging.info('Running ablation: balancing the sparsity rate')

      num_blocks = 32
      for i in range(1,num_blocks):
        sparsity = i / num_blocks
        ablation_balancing_sparsity_ratio(args.model, sparsity, calibration_dataset_2ssp, wikitext_input_ids, seed=args.seed, cache_dir=args.cache_dir)


if __name__ == "__main__":
  main()
