# Adapted from ALMA github repo: https://github.com/fe1ixxu/ALMA

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import argparse
import pandas as pd
from tqdm import tqdm

device = "auto"

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--dataset_path', type=str, default="../dataset/alpaca.csv", help='Path to the dataset')
    parser.add_argument('--target_lang', type=str, default="Chinese", help='target language be translated to')
    parser.add_argument('--target_lang_id', type=str, default="zh", help='id of the target language')
    return parser.parse_args()



def translate(dataset_path, target_lang, target_lang_id): 

    # Choose appropriate model
    # ISO 639 Language Codes
    GROUP2LANG = {
        1: ["da", "nl", "de", "is", "no", "sv", "af"],
        2: ["ca", "ro", "gl", "it", "pt", "es"],
        3: ["bg", "mk", "sr", "uk", "ru"],
        4: ["id", "ms", "th", "vi", "mg", "fr"],
        5: ["hu", "el", "cs", "pl", "lt", "lv"],
        6: ["ka", "zh", "ja", "ko", "fi", "et"],
        7: ["gu", "hi", "mr", "ne", "ur"],
        8: ["az", "kk", "ky", "tr", "uz", "ar", "he", "fa"],
    }
    LANG2GROUP = {lang: str(group) for group, langs in GROUP2LANG.items() for lang in langs}
    group_id = LANG2GROUP[target_lang_id]

    # Load model
    model = AutoModelForCausalLM.from_pretrained(f"haoranxu/X-ALMA-13B-Group{group_id}", torch_dtype=torch.float16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(f"haoranxu/X-ALMA-13B-Group{group_id}", padding_side='left')

    # Load dataset
    df = pd.read_csv(dataset_path) # alpaca.csv

    # Translation
    def translate_text(prompts, tokenizer, model, target_lang):
        with torch.no_grad():
            translations = []
            for prompt in tqdm(prompts):
                chat_style_prompt = [{"role": "user", "content": f"Translate this from English to {target_lang}:\nEnglish: {prompt} \n{target_lang}:"}]
                encoded_prompt = tokenizer.apply_chat_template(chat_style_prompt, tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer(encoded_prompt, return_tensors="pt", padding=True, max_length=150, truncation=True).input_ids.cuda()
                generated_ids = model.generate(input_ids=input_ids, num_beams=5, max_new_tokens=100, do_sample=True, temperature=0.6, top_p=0.9)
                translation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                translation = translation.split("[/INST]")[1]
                translations.append(translation)
        return translations
    
    harmless_translation = translate_text(df['Safe Prompt'].tolist(), tokenizer, model, target_lang)
    harmful_translation = translate_text(df['Prompt'].tolist(), tokenizer, model, target_lang)

    # Update the DataFrame with the translated text
    df['Safe Prompt'] = harmless_translation
    df['Prompt'] = harmful_translation
    
    output_path = f"../dataset/alpaca_{target_lang}.csv"
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    #args = parse_arguments()
    #translate(args.dataset_path, args.target_lang, args.target_lang_id)
    #translate("../dataset/alpaca.csv", "Chinese", "zh")
    translate("../dataset/alpaca.csv", "German", "de")
    translate("../dataset/alpaca.csv", "French", "fr")
    translate("../dataset/alpaca.csv", "Italian", "it")
    