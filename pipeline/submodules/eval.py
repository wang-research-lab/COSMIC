import argparse
import json
import os
import re
from tqdm import tqdm
import torch
import gc
import pdb

from data_utils import BASE_PROMPT
from utils import load_lora_model_and_tokenizer, generate_completions
from transformers import GenerationConfig

def get_prompts_response(data_path):
    with open(data_path, 'r') as f:  
        dataset = json.load(f)
    prompts = [BASE_PROMPT.format(instruction=example['instruction']) for example in dataset]
    correct_answer = [str(example["answer"]) for example in dataset]
    return prompts, correct_answer

def extract_answer(dataset, sentence: str) -> str:
    response = sentence['response'].lower().strip()

    patterns = {
        'boolq': ['true', 'false'],
        'piqa': ['solution1', 'solution2'],
        'social_i_qa': ['answer1', 'answer2', 'answer3', 'answer4','answer5'],
        'ARC-Challenge': ['answer1', 'answer2', 'answer3', 'answer4','answer5'],
        'ARC-Easy': ['answer1', 'answer2', 'answer3', 'answer4','answer5'],
        'openbookqa': ['answer1', 'answer2', 'answer3', 'answer4','answer5'],
        'hellaswag': ['ending1','ending2','ending3','ending4']
    }

    for pattern in patterns.get(dataset, []):
        if re.search(r'\b' + pattern + r'\b', response):
            return pattern
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to model", required=True)
    parser.add_argument("--lora_path", type=str, help="Path to lora adapters", required=True)
    parser.add_argument("--dataset_path", type=str, help="Path to dataset", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    dataset_name = os.path.basename(args.dataset_path).split('.')[0]

    prompts, correct_answer = get_prompts_response(args.dataset_path)
    model, tokenizer = load_lora_model_and_tokenizer(args.model_path, args.lora_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        bos_token_id=model.config.bos_token_id,
    )

    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=args.batch_size,
        generation_config = generation_config
    )

    save_outputs = []
    correct = 0
    count = 0
    for correct_output, output in zip(correct_answer, outputs):
        predict = extract_answer(dataset_name, output)
        if predict == "":
            continue
        if correct_output.lower() == predict.lower():
            correct += 1
        count += 1

        save_outputs.append({
            'raw_output': output,
            'prediction': predict,
            'correct_output': correct_output,
        })
    
    weighted_acc = correct/len(prompts)
    print(f"{dataset_name} Accuracy: {weighted_acc * 100:.1f}%, Total: {len(prompts)}")

    print(f"\n{dataset_name} Results:")
    print(f"Accuracy: {weighted_acc * 100:.1f}%")
    print(f"Total samples: {len(prompts)}")
    
    # Save results with dataset name
    with open(os.path.join(args.output_dir, f"{dataset_name}_predictions.jsonl"), "w") as f:
        for example in save_outputs:
            f.write(json.dumps(example) + "\n")

    del model
    gc.collect()
    torch.cuda.empty_cache()