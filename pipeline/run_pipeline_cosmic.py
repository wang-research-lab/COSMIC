import torch
import random
import json
import os
import argparse
import numpy as np
from tqdm import tqdm

from dataset.load_dataset import load_dataset_split, load_dataset


from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import (
    add_hooks, 
    get_affine_activation_addition_input_pre_hook,  
    get_affine_direction_ablation_input_pre_hook, 
    get_all_linear_direction_ablation_hooks,
    get_linear_activation_addition_input_pre_hook,  
    get_linear_direction_ablation_input_pre_hook
)           

from pipeline.submodules.generate_directions_cosmic import generate_directions
from pipeline.submodules.select_direction_cosmic import select_direction
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss
from pipeline.submodules.evaluate_reasoning import evaluate_gpqa_and_arc, evaluate_truthful_qa
from pipeline.submodules.completions_helper import (
    generate_and_save_completions_for_dataset,
    evaluate_completions_and_save_results_for_dataset
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run COSMIC pipeline.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--affine_steering', type=str, choices=["true", "false"], default="true", help="Use affine (true) or linear (false) steering")
    args = parser.parse_args()

    # Manually convert string to actual bool
    args.affine_steering = args.affine_steering.lower() == "true"
    return args


def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), cfg.n_train)
    harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
    harmful_val = random.sample(load_dataset_split(harmtype='harmful', split='val', instructions_only=True), cfg.n_val)
    harmless_val = random.sample(load_dataset_split(harmtype='harmless', split='val', instructions_only=True), cfg.n_val)
    return harmful_train, harmless_train, harmful_val, harmless_val

def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train):
    """Generate and save candidate directions."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_directions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'))

    mean_diffs, harmless_mean = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"))

    torch.save(mean_diffs, os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt'))

    return mean_diffs, harmless_mean

def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions, harmless_mean, layers_to_evaluate, affine_steering):
    """Select and save the direction."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'select_direction')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'select_direction'))

    pos, layer, direction, harmless_reference = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        harmless_mean,
        layers_to_evaluate,
        affine_steering = affine_steering,
        artifact_dir=os.path.join(cfg.artifact_path(), "select_direction")
    )
    
    if os.path.exists(f'{cfg.artifact_path()}/direction_metadata.json'):
        with open(f'{cfg.artifact_path()}/direction_metadata.json', "r") as f:
            saved_metadata = json.load(f)
    
    with open(f'{cfg.artifact_path()}/direction_metadata.json', "w") as f:
        json.dump({"pos": pos, 
                   "layer": layer}, f, indent=4)

    torch.save(direction, f'{cfg.artifact_path()}/direction.pt')

    return pos, layer, direction, harmless_reference

def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens, batch_size = 64)
    
    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)

def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    """Evaluate completions and save results for a dataset."""
    with open(os.path.join(cfg.artifact_path(), f'completions/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
    )

    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_evaluations.json', "w") as f:
        json.dump(evaluation, f, indent=4)

def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    """Evaluate loss on datasets."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'loss_evals')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'loss_evals'))

    on_distribution_completions_file_path = os.path.join(cfg.artifact_path(), f'completions/harmless_baseline_completions.json')

    loss_evals = evaluate_loss(model_base, fwd_pre_hooks, fwd_hooks, batch_size=cfg.ce_loss_batch_size, n_batches=cfg.ce_loss_n_batches, completions_file_path=on_distribution_completions_file_path)

    with open(f'{cfg.artifact_path()}/loss_evals/{intervention_label}_loss_eval.json', "w") as f:
        json.dump(loss_evals, f, indent=4)

def compute_and_save_similarity(model, cfg, harmful_data, harmless_data, tokenize_fn, fwd_pre_hooks=[], fwd_hooks=[], batch_size=32, system_prompt = None,fraction = 0.10):
    """
    Computes cosine similarity between mean activations of harmful and harmless data for each layer
    and returns the layers ranked by the least similarities.
    """
    
    num_hidden_layers = model.config.num_hidden_layers  # Total layers

    if not os.path.exists(os.path.join(cfg.artifact_path(), 'select_direction')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'select_direction'))

    save_path = f'{cfg.artifact_path()}/select_direction/COSMIC_evaluation_layers.json'
    

    # Initialize tensors to store outputs
    harmful_outputs = torch.zeros((len(harmful_data), num_hidden_layers, model.config.hidden_size), device=model.device)
    harmless_outputs = torch.zeros((len(harmless_data), num_hidden_layers, model.config.hidden_size), device=model.device)
    

    progress = tqdm(total=2, desc="Finding COSMIC Evaluation Layers")

    for category, data, outputs in zip(["harmful", "harmless"], [harmful_data, harmless_data], [harmful_outputs, harmless_outputs]):
        for i in range(0, len(data), batch_size):
            batch_instructions = data[i:i+batch_size]
            tokenized = tokenize_fn(instructions=batch_instructions, system=system_prompt)

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                model_output = model(
                    input_ids=tokenized.input_ids.to(model.device),
                    attention_mask=tokenized.attention_mask.to(model.device),
                    output_hidden_states=True
                )

            hidden_states = model_output.hidden_states[1:]  # Exclude embedding layer
            for j, layer_output in enumerate(hidden_states):
                outputs[i:i+batch_size, j, :] = layer_output[:, -1, :]

        progress.update(1)

    progress.close()

    # Compute mean activations per layer
    mean_harmful = harmful_outputs.mean(dim=0)  # Shape: (eval_layers, hidden_size)
    mean_harmless = harmless_outputs.mean(dim=0)  # Shape: (eval_layers, hidden_size)
    
    # Normalize mean activations
    normalized_harmful = mean_harmful / torch.norm(mean_harmful, dim=-1, keepdim=True)
    normalized_harmless = mean_harmless / torch.norm(mean_harmless, dim=-1, keepdim=True)
    
    # Compute cosine similarity between harmful and harmless activations per layer
    similarities = torch.sum(normalized_harmful * normalized_harmless, dim=-1).cpu().numpy()

    # Rank layers by least similarities
    ranked_layers = np.argsort(similarities).tolist()  # Convert to list here

    with open(save_path, "w") as f:
        json.dump({"ranked_layers": ranked_layers}, f, indent=4)

    return ranked_layers[:int(fraction * num_hidden_layers)]

def run_pipeline(model_path, affine_steering):
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path) + f"-cosmic-{'ace' if affine_steering else 'lce'}"
    cfg = Config(model_alias=model_alias, model_path=model_path)
    
    model_base = construct_model_base(cfg.model_path)
    
    # Load and sample datasets
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)

    # 1. Generate candidate refusal directions
    candidate_directions, harmless_mean = generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train)
    
    layers_to_evaluate = compute_and_save_similarity(model_base.model,
                                                    cfg, 
                                                    harmful_train, 
                                                    harmless_train,
                                                    model_base.tokenize_instructions_fn, 
                                                    fwd_pre_hooks=[], 
                                                    fwd_hooks=[])

    # 2. Select the most effective refusal direction
    pos, layer, pre_layer_direction, pre_layer_reference  = select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions, harmless_mean, layers_to_evaluate, affine_steering)


    #3.construct hooks
    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []

    if affine_steering:
        ablation_fwd_pre_hooks = [(model_base.model_block_modules[layer], get_affine_direction_ablation_input_pre_hook(direction=pre_layer_direction, reference = pre_layer_reference))]
        ablation_fwd_hooks = []

        coeff = torch.Tensor([1.0])
        actadd_refusal_pre_hooks = [(model_base.model_block_modules[layer], get_affine_activation_addition_input_pre_hook(direction=pre_layer_direction, coeff = coeff, reference = pre_layer_reference))]
        actadd_refusal_fwd_hooks = []
        
    else: 
        ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_linear_direction_ablation_hooks(model_base, pre_layer_direction)

        actadd_refusal_pre_hooks, actadd_refusal_fwd_hooks = [(model_base.model_block_modules[layer], 
                                                get_linear_activation_addition_input_pre_hook(vector=pre_layer_direction, 
                                                coeff=torch.Tensor([1.0])))], []
        
    
    #4a. Generate compeltions on harmful eval dataset 
    harmful_test = random.sample(load_dataset_split(harmtype='harmful', split='test'), cfg.n_test)
    generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'harmful', dataset = harmful_test)
    generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', 'harmful', dataset = harmful_test)

    #4a. Generate compeltions on harmless eval dataset 
    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)
    generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'harmless', dataset=harmless_test)
    generate_and_save_completions_for_dataset(cfg, model_base, actadd_refusal_pre_hooks, actadd_refusal_fwd_hooks, 'actadd', 'harmless', dataset=harmless_test)

    """evaluate_gpqa_and_arc(cfg, model_base, 
                           ablation_fwd_pre_hooks, 
                           ablation_fwd_hooks, 
                           exclude_base = True,
                           batch_size = 32)
    evaluate_truthful_qa(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, batch_size = 8)"""
    
    

    #we load in llamaguard now for evals
    #so we're deleting the model to spare your vram
    del model_base
    torch.cuda.empty_cache() 

    # 5. Evaluate completions and save results on harmless evaluation dataset
    evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)
    evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)


    # 5b. Evaluate completions and save results on harmful evaluation datasets
    evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', "harmful", eval_methodologies=cfg.jailbreak_eval_methodologies)
    evaluate_completions_and_save_results_for_dataset(cfg, 'ablation', "harmful", eval_methodologies=cfg.jailbreak_eval_methodologies)
    

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, affine_steering = args.affine_steering)
    torch.cuda.empty_cache()
