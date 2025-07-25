import torch
import random
import json
import os
import argparse
import numpy as np

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks, get_direction_ablation_input_pre_hook, get_direction_ablation_output_hook, get_activation_addition_input_post_hook

from pipeline.submodules.generate_directions_cosmic import generate_directions
from pipeline.submodules.select_direction_cosmic import select_direction, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss
from pipeline.submodules.evaluate_reasoning import evaluate_gpqa_and_arc, evaluate_truthful_qa
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

    pos, layer, direction, harmless_reference, selected_coeffs = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        harmless_mean,
        artifact_dir=os.path.join(cfg.artifact_path(), "select_direction")
    )

    with open(f'{cfg.artifact_path()}/direction_metadata.json', "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(direction, f'{cfg.artifact_path()}/direction.pt')

    return pos, layer, direction, harmless_reference, selected_coeffs

def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens)
    
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

def run_pipeline(model_path):
    
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path) + "-ace"
    cfg = Config(model_alias=model_alias, model_path=model_path)

    model_base = construct_model_base(cfg.model_path)

    # Load and sample datasets
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)
    
    # Filter datasets based on refusal scores
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val)

    # 1. Generate candidate refusal directions
    candidate_directions, harmless_mean = generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train)
    
    # 2. Select the most effective refusal direction
    #pos, layer, direction, harmless_reference, coefficients  = select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions, harmless_mean)

    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []

    pos = -1
    if "gemma" in model_path.lower():
        layer = 14
    elif "llama" in model_path.lower():
        layer = 15
    elif "qwen" in model_path.lower():
        layer = 20
    else:
        raise AssertionError("unknown model")

    direction = torch.stack((candidate_directions[0][pos, layer],candidate_directions[1][pos, layer], candidate_directions[2][pos, layer], candidate_directions[3][pos, layer]), dim=0)
    harmless_reference = torch.stack((harmless_mean[0][pos, layer], harmless_mean[1][pos, layer], harmless_mean[2][pos, layer], harmless_mean[3][pos, layer]), dim=0)

    pre_layer_direction = direction[0]
    post_attn_direction = direction[1]
    post_mlp_direction = direction[2]
    post_layer_direction = direction[3]

    pre_layer_reference = harmless_reference[0]
    post_attn_reference = harmless_reference[1]
    post_mlp_reference = harmless_reference[2]
    post_layer_reference = harmless_reference[3]

    negative_actadd_coeff = 0
    positive_actadd_coeff = 1

    coeff = torch.Tensor([0])

    ablation_fwd_pre_hooks = []#[(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=pre_layer_direction, coeff = coeff, reference = pre_layer_reference))]
    #ablation_fwd_hooks = [(model_base.model_post_attn_modules[layer], get_direction_ablation_output_hook(direction=post_attn_direction, coeff = coeff, reference = post_attn_reference))]
    ablation_fwd_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_output_hook(direction=post_layer_direction, coeff = coeff, reference = post_layer_reference))]


    coeff = torch.Tensor([negative_actadd_coeff])
    actadd_fwd_pre_hooks = []#[(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(direction=pre_layer_direction, coeff = coeff, reference = pre_layer_reference))]
    #actadd_fwd_hooks = [(model_base.model_post_attn_modules[layer], get_activation_addition_input_post_hook(direction=post_attn_direction, coeff = coeff, reference = post_attn_reference))]
    actadd_fwd_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_post_hook(direction=post_layer_direction, coeff = coeff, reference = post_layer_reference))]

    """evaluate_gpqa_and_arc(cfg, model_base, 
                           ablation_fwd_pre_hooks, 
                           ablation_fwd_hooks, 
                           actadd_fwd_pre_hooks, 
                           actadd_fwd_hooks,
                           batch_size = 8)
    
    evaluate_truthful_qa(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, actadd_fwd_pre_hooks, actadd_fwd_hooks, batch_size = 8)"""

    # 3a. Generate and save completions on harmful evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', dataset_name)
        generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', dataset_name)
        generate_and_save_completions_for_dataset(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd', dataset_name)

    # 4a. Generate and save completions on harmless evaluation dataset
    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)

    generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'harmless', dataset=harmless_test)
    
    coeff = torch.Tensor([positive_actadd_coeff])
    actadd_refusal_pre_hooks = []#[(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(direction=pre_layer_direction, coeff = coeff, reference = pre_layer_reference))]
    #actadd_refusal_hooks = [(model_base.model_post_attn_modules[layer], get_activation_addition_input_post_hook(direction=post_attn_direction, coeff = coeff, reference = post_attn_reference))]
    actadd_refusal_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_post_hook(direction=post_layer_direction, coeff = coeff, reference = post_layer_reference))]

    generate_and_save_completions_for_dataset(cfg, model_base, actadd_refusal_pre_hooks, actadd_refusal_hooks, 'actadd', 'harmless', dataset=harmless_test)

    """# 5. Evaluate loss on harmless datasets
    evaluate_loss_for_datasets(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline')
    evaluate_loss_for_datasets(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation')
    evaluate_loss_for_datasets(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd')"""

    #we load in llamaguard now for evals
    #so we're deleting the model to spare your vram
    del model_base
    torch.cuda.empty_cache() 

    # 4b. Evaluate completions and save results on harmless evaluation dataset
    evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)
    evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)

    # 3b. Evaluate completions and save results on harmful evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, 'ablation', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
    


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path)
    torch.cuda.empty_cache()
