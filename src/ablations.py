from utilities import (
  get_calibration,
  loadModel,
  set_seed,
  get_mlp_inputs_outputs,
  maskModel,
  get_mlp_hidden_state,
  prune_mlp,
  second_stage_attention
)
from evaluation import evaluate_perplexity
from pruning import two_stage_2ssp

from tqdm import tqdm
from types import MethodType
import gc
import torch
import time


def ablation_calibration_dataset(
  model_name,
  tokenizer,
  sparsity,
  dataset_c4,
  wikitext_input_ids,
  calibration_sizes,
  seq_len=2048,
  seed=0,
  method="one_stage_2ssp",
  cache_dir=None
):
  print(f"************* Test at sparsity {sparsity} - {method} *************")

  seeds = [0, 1, 2, 3, 4]
  for cs in calibration_sizes:
    for seed in seeds:
      model = loadModel(model_name, cache_dir)

      set_seed(seed)
      calibration_dataset = get_calibration(
        dataset_c4, tokenizer, num_samples=cs, seq_len=seq_len, seed=seed
      )

      # Pruning
      start_time = time.time()
      if method == "one_stage_2ssp":
        model = one_stage_2ssp(model, calibration_dataset, sparsity)
      elif method == "2ssp":
        model = two_stage_2ssp(model, calibration_dataset, sparsity)
      else:
        print("ERROR: use only [one_stage_2ssp, 2ssp]")
        exit(1)

      end_time = time.time()

      # Evaluation
      ppl = evaluate_perplexity(model, wikitext_input_ids, seq_len=2048)
      print(f"Pruning Time: {end_time - start_time} s")
      print(f"Calibration size {cs}, Seed {seed}, perplexity {ppl}")

      # Free memory for model reload at next iteration
      del model
      gc.collect()
      torch.cuda.empty_cache()




"""
2SSP (first stage only)

Implements the first stage of 2SSP. Prunes only the MLP submodules across
transformer blocks based on L2 norms of their hidden states, using a calibration
dataset to guide the pruning process.

Args:
  model (torch.nn.Module): The transformer model to prune. 
  calibration_dataset (Iterable[torch.Tensor]): A dataset of samples used 
                        to calculate hidden state norms for pruning decisions.
  pruning_rate (float): The overall proportion of parameters to prune, 
                        relative to the total number of parameters

Returns:
  torch.nn.Module: The pruned transformer model, with MLP submodules pruned 
"""
@torch.no_grad()
def one_stage_2ssp(model, calibration_dataset, pruning_rate):
    
    layers = model.model.layers
    num_blocks = len(layers)

    print("======================")
    model_total_params = sum(p.numel() for p in model.parameters())
    main_model_total_params = sum(p.numel() for p in model.model.layers.parameters())
    attn_total_params = sum(p.numel() for p in model.model.layers[0].self_attn.parameters())
    mlp_total_params = sum(p.numel() for p in model.model.layers[0].mlp.parameters())
    print(f"[Original model] Full number of parameters = {model_total_params}")
    print(f"[Original model] Main model number of parameters = {main_model_total_params}")
    print(f"Attention parameters (one block): {attn_total_params}")
    print(f"MLP parameters (one block): {mlp_total_params}")
    print("======================")

    mlp_pruning_rate = pruning_rate * (main_model_total_params / (num_blocks * mlp_total_params))

    mlp_intermediate_size = model.config.intermediate_size
    num_preserve_mlp = int(round(mlp_intermediate_size * (1 - mlp_pruning_rate)))

    average_norms = [0] * num_blocks
    for sample in tqdm(calibration_dataset, total=len(calibration_dataset), desc="Collecting MLP hidden state"):
      hidden_states_ci = get_mlp_hidden_state(model, sample)

      for li in range(num_blocks):
        norm_ci_li = hidden_states_ci[li].norm(dim=0, p=2)
        average_norms[li] += norm_ci_li
    
    for li in range(num_blocks):
      average_norms[li] /= len(calibration_dataset)
      _, top_indices = torch.topk(average_norms[li], num_preserve_mlp)
      mask = torch.ones_like(average_norms[li])
      mask[top_indices] = 0

      prune_mlp(model, mask, li)  

    return model


@torch.no_grad()
def prune_mlp_inverted(model, mask_inputs, mask_outputs, layer_i):
  preserve_mask_inputs = torch.where(mask_inputs == 0)[0]
  preserve_mask_outputs = torch.where(mask_outputs == 0)[0]

  layer = model.model.layers[layer_i]

  if model.config.model_type in ("llama", "mistral", "qwen2"):
    layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[
      :, preserve_mask_inputs
    ]
    layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[
      :, preserve_mask_inputs
    ]
    layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[
      preserve_mask_outputs
    ]
  else:
    print(f"Error: {model.config.model_type} is not supported")
    exit(0)

  def pruned_forward(self, hidden_state):
    hidden_state = hidden_state[:, :, preserve_mask_inputs]

    down_proj = self.down_proj(
      self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
    )

    down_proj_reshaped = torch.zeros(
      (down_proj.size(0), down_proj.size(1), model.config.hidden_size),
      device=model.device,
      dtype=down_proj.dtype,
    )
    down_proj_reshaped[:, :, preserve_mask_outputs] = down_proj

    return down_proj_reshaped

  layer.mlp.forward = MethodType(pruned_forward, layer.mlp)



@torch.no_grad()
def two_stage_2ssp_inverted(
  model, calibration_dataset, pruning_rate, num_attn_submodules_to_prune=None
):
  layers = model.model.layers
  num_layers = len(layers)

  print("======================")
  model_total_params = sum(p.numel() for p in model.parameters())
  main_model_total_params = sum(p.numel() for p in model.model.layers.parameters())
  attn_total_params = sum(
    p.numel() for p in model.model.layers[0].self_attn.parameters()
  )
  mlp_total_params = sum(p.numel() for p in model.model.layers[0].mlp.parameters())
  print(f"[Original model] Full number of parameters = {model_total_params}")
  print(f"[Original model] Main model number of parameters = {main_model_total_params}")
  print(f"Attention parameters (one block): {attn_total_params}")
  print(f"MLP parameters (one block): {mlp_total_params}")
  print("======================")

  if num_attn_submodules_to_prune is None:
    num_attn_submodules_to_prune = round(
      num_layers * pow(pruning_rate, (mlp_total_params / attn_total_params) / 1.5)
    )
  print(f"Pruning {num_attn_submodules_to_prune} attention submodules")

  if (
    num_attn_submodules_to_prune * attn_total_params
  ) / main_model_total_params > pruning_rate:
    print("Exceeded pruning parameters number")
    return False

  if (
    num_attn_submodules_to_prune * attn_total_params + num_layers * mlp_total_params
  ) / main_model_total_params < pruning_rate:
    print(
      f"Unable to reach the target sparsity rate with only {num_attn_submodules_to_prune} pruned attention submodules"
    )
    return False

  # 1. Pruning the MLP
  mlp_pruning_rate = pruning_rate * (
    main_model_total_params / (num_layers * mlp_total_params)
  )

  model_hidden_size = model.config.hidden_size
  num_preserve_hidden = int(round(model_hidden_size * (1 - mlp_pruning_rate)))

  average_norms_inputs = [0] * num_layers
  average_norms_outputs = [0] * num_layers
  for sample in tqdm(
    calibration_dataset,
    total=len(calibration_dataset),
    desc="Collecting MLP hidden state",
  ):
    mlp_inputs_ci, mlp_outputs_ci = get_mlp_inputs_outputs(model, sample)

    for li in range(num_layers):
      norm_ci_li_inputs = mlp_inputs_ci[li].norm(dim=0, p=2)
      norm_ci_li_outputs = mlp_outputs_ci[li].norm(dim=0, p=2)
      average_norms_inputs[li] += norm_ci_li_inputs
      average_norms_outputs[li] += norm_ci_li_outputs

  for li in range(num_layers):
    average_norms_inputs[li] /= len(calibration_dataset)
    average_norms_outputs[li] /= len(calibration_dataset)
    _, top_indices_inputs = torch.topk(average_norms_inputs[li], num_preserve_hidden)
    _, top_indices_outputs = torch.topk(average_norms_outputs[li], num_preserve_hidden)
    mask_inputs = torch.ones_like(average_norms_inputs[li])
    mask_outputs = torch.ones_like(average_norms_inputs[li])
    mask_inputs[top_indices_inputs] = 0
    mask_outputs[top_indices_outputs] = 0

    prune_mlp_inverted(model, mask_inputs, mask_outputs, li)

  # 2. Pruning the Attention submodules
  num_calib_blockpruner = 1
  calibration_input_ids = torch.cat(calibration_dataset[:num_calib_blockpruner], dim=1)

  attnMask, mlpMask = second_stage_attention(
    model,
    num_prune=num_attn_submodules_to_prune,
    calibration_input_ids=calibration_input_ids,
  )
  maskModel(model, attnMask=attnMask, mlpMask=mlpMask)

  return model


# 2SSP with two stages using L1 norm instead of L2 norm
@torch.no_grad()
def two_stage_2ssp_l1_norm(
  model, calibration_dataset, pruning_rate, num_attn_submodules_to_prune=None
):
  layers = model.model.layers
  num_layers = len(layers)

  print("======================")
  model_total_params = sum(p.numel() for p in model.parameters())
  main_model_total_params = sum(p.numel() for p in model.model.layers.parameters())
  attn_total_params = sum(
    p.numel() for p in model.model.layers[0].self_attn.parameters()
  )
  mlp_total_params = sum(p.numel() for p in model.model.layers[0].mlp.parameters())
  print(f"[Original model] Full number of parameters = {model_total_params}")
  print(f"[Original model] Main model number of parameters = {main_model_total_params}")
  print(f"Attention parameters (one block): {attn_total_params}")
  print(f"MLP parameters (one block): {mlp_total_params}")
  print("======================")

  if num_attn_submodules_to_prune is None:
    num_attn_submodules_to_prune = round(
      num_layers * pow(pruning_rate, (mlp_total_params / attn_total_params) / 1.5)
    )
  print(f"Pruning {num_attn_submodules_to_prune} attention submodules")

  if (
    num_attn_submodules_to_prune * attn_total_params
  ) / main_model_total_params > pruning_rate:
    print("Exceeded pruning parameters number")
    return False

  if (
    num_attn_submodules_to_prune * attn_total_params + num_layers * mlp_total_params
  ) / main_model_total_params < pruning_rate:
    print(
      f"Unable to reach the target sparsity rate with only {num_attn_submodules_to_prune} pruned attention submodules"
    )
    return False

  # 1. Pruning the MLP (FFN Pruner)
  parameters_pruned_for_attention = num_attn_submodules_to_prune * attn_total_params
  target_parameters_to_prune = int(round(pruning_rate * main_model_total_params))
  mlp_params_to_prune = int(
    round((target_parameters_to_prune - parameters_pruned_for_attention) / num_layers)
  )  # number of remaining parameters to prune in each single mlp block
  mlp_pruning_rate = mlp_params_to_prune / mlp_total_params

  mlp_hidden_size = model.config.intermediate_size
  num_preserve_mlp = int(round(mlp_hidden_size * (1 - mlp_pruning_rate)))

  average_norms = [0] * num_layers
  for sample in tqdm(
    calibration_dataset,
    total=len(calibration_dataset),
    desc="Collecting MLP hidden state",
  ):
    hidden_states_ci = get_mlp_hidden_state(model, sample)

    for li in range(num_layers):
      norm_ci_li = hidden_states_ci[li].norm(dim=0, p=1)
      average_norms[li] += norm_ci_li

  for li in range(num_layers):
    average_norms[li] /= len(calibration_dataset)
    _, top_indices = torch.topk(average_norms[li], num_preserve_mlp)
    mask = torch.ones_like(average_norms[li])
    mask[top_indices] = 0

    prune_mlp(model, mask, li)

  # 2. Pruning the Attention submodules
  num_calib_blockpruner = 1
  calibration_input_ids = torch.cat(calibration_dataset[:num_calib_blockpruner], dim=1)

  attnMask, mlpMask = second_stage_attention(
    model,
    num_prune=num_attn_submodules_to_prune,
    calibration_input_ids=calibration_input_ids,
  )
  maskModel(model, attnMask=attnMask, mlpMask=mlpMask)

  return model


def ablation_balancing_sparsity_ratio(
  model_name, sparsity, calibration_dataset_ffn_pruner, wikitext_input_ids, seed=0, cache_dir=None
):
  num_blocks = 1
  num_attn_submodules_to_prune = 0

  while num_attn_submodules_to_prune < num_blocks:
    model = loadModel(model_name, cache_dir)
    num_blocks = len(model.model.layers)

    # Pruning
    set_seed(seed)
    model = two_stage_2ssp(
      model, calibration_dataset_ffn_pruner, sparsity, num_attn_submodules_to_prune
    )

    # Evaluation
    if model != False:
      ppl = evaluate_perplexity(model, wikitext_input_ids, seq_len=2048)
      print(
        f"When pruning {num_attn_submodules_to_prune} attention submodules @ {sparsity}, perplexity is {ppl}"
      )

    # Free memory for reload
    del model
    gc.collect()
    torch.cuda.empty_cache()

    num_attn_submodules_to_prune += 1
