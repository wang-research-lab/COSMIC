import torch
import random
import json
import os
import argparse
import numpy as np
import pandas as pd

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks, get_direction_ablation_input_pre_hook, get_direction_ablation_output_hook, get_activation_addition_input_post_hook

from pipeline.submodules.generate_directions_cosmic import generate_directions
from pipeline.submodules.select_direction_2 import select_direction, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss
from pipeline.submodules.evaluate_reasoning import evaluate_gsm8k_and_arc, evaluate_truthful_qa
from pipeline.utils.hook_utils import add_hooks
import pdb

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    return parser.parse_args()

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

def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)
    
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

def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions, harmless_mean):
    """Select and save the direction."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'select_direction')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'select_direction'))

    pos, layer, direction, harmless_reference, ablation_enhance_coeff = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        harmless_mean,
        artifact_dir=os.path.join(cfg.artifact_path(), "select_direction")
    )

    
    with open(f'{cfg.artifact_path()}/direction_metadata.json', "w") as f:
        json.dump({"pos": pos, 
                   "layer": layer,
                   "ablation_enhance_coeff" : ablation_enhance_coeff}, f, indent=4)

    torch.save(direction, f'{cfg.artifact_path()}/direction.pt')

    return pos, layer, direction, harmless_reference, ablation_enhance_coeff

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


def compute_and_save_similarity(model, alias, harmful_data, harmless_data, tokenize_fn, fwd_pre_hooks=[], fwd_hooks=[], batch_size=32, output_file="cosine_similarity.csv"):
    """
    Computes cosine similarity between mean activations of harmful and harmless data for each layer and saves results as a CSV.
    """
    
    num_hidden_layers = model.config.num_hidden_layers  # Total layers

    # Initialize tensors to store outputs
    harmful_outputs = torch.zeros((len(harmful_data), num_hidden_layers, model.config.hidden_size), device=model.device)
    harmless_outputs = torch.zeros((len(harmless_data), num_hidden_layers, model.config.hidden_size), device=model.device)
    
    for category, data, outputs in zip(["harmful", "harmless"], [harmful_data, harmless_data], [harmful_outputs, harmless_outputs]):
        for i in range(0, len(data), batch_size):
            batch_instructions = data[i:i+batch_size]
            tokenized = tokenize_fn(instructions=batch_instructions)
            
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                model_output = model(
                    input_ids=tokenized.input_ids.to(model.device),
                    attention_mask=tokenized.attention_mask.to(model.device),
                    output_hidden_states=True
                )
            
            hidden_states = model_output.hidden_states[1:]  # Exclude embedding layer
            for j, layer_output in enumerate(hidden_states, start=0):
                outputs[i:i+batch_size, j, :] = layer_output[:, -1, :]
    
    # Compute mean activations per layer
    mean_harmful = harmful_outputs.mean(dim=0)  # Shape: (eval_layers, hidden_size)
    mean_harmless = harmless_outputs.mean(dim=0)  # Shape: (eval_layers, hidden_size)
    
    # Normalize mean activations
    normalized_harmful = mean_harmful / torch.norm(mean_harmful, dim=-1, keepdim=True)
    normalized_harmless = mean_harmless / torch.norm(mean_harmless, dim=-1, keepdim=True)
    
    # Compute cosine similarity between harmful and harmless activations per layer
    similarities = torch.sum(normalized_harmful * normalized_harmless, dim=-1).cpu().numpy()
    
    output_file = alias + "_" + output_file
    # Save results to CSV
    df = pd.DataFrame({"Layer": range(num_hidden_layers), "Cosine_Similarity": similarities})
    df.to_csv(output_file, index=False)
    print(f"Saved cosine similarity results to {output_file}")



def run_pipeline(model_path):
    
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)
    
    model_base = construct_model_base(cfg.model_path)


    # Load and sample datasets
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)
    
    # Filter datasets based on refusal scores
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val)


    compute_and_save_similarity(model_base.model, model_alias, harmful_train, harmless_train, model_base.tokenize_instructions_fn)

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path)
    torch.cuda.empty_cache()
