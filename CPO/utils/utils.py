from typing import Literal
from math import isclose
from transformers import training_args
from datasets import concatenate_datasets

LANGUAGE_IDS = {
    "en": "Engelsk",
    "nb": "Bokmål",
    "nn": "Nynorsk"
}


def get_prompt(source_lang: str, target_lang: str, ex: dict[str, str], style: Literal["instruct", "basic"]):
    match style:
        case "instruct":
            text = f"""{LANGUAGE_IDS[source_lang]}: {ex[source_lang]}
{LANGUAGE_IDS[target_lang]}:"""
            
        case "basic":
            text = f"""Oversett dette fra {LANGUAGE_IDS[source_lang]} til {LANGUAGE_IDS[target_lang]}: 
{LANGUAGE_IDS[source_lang]}: {ex[source_lang]}
{LANGUAGE_IDS[target_lang]}:"""
            
        case _:
            raise ValueError(f"Invalid style ({style}).")

    return text
    


# Inspired by https://github.com/fe1ixxu/ALMA/blob/master/utils/utils.py
def preprocess_cpo_data(train_raw_data, pairs, tokenizer, args: "argparse.Namespace"):
    
    def get_chosen_reject(example: dict[str, str], target_lang: str, qe: bool = False) -> tuple[str, str]:
        """Given an example entry and target lang, return the chosen and rejected sentences"""
        # If using XAlma preference dataset
        if "chosen" in example and "reject" in example:
            # Add newline at the end to make the module prdouce until newline character
            return f"{example['chosen']}\n", f"{example['reject']}\n"
        
        high_score_key = f"high_{target_lang}_{args.cpo_scorer}"
        low_score_key = f"low_{target_lang}_{args.cpo_scorer}"
        ref_score_key = f"ref_{target_lang}_{args.cpo_scorer}"

        high_output_key = f"high_{target_lang}"
        low_output_key = f"low_{target_lang}"
        ref_output_key = target_lang

        # Defining the sentences and their scores
        sentences = [example[ref_output_key], example[high_output_key], example[low_output_key]]
        scores = [example[ref_score_key], example[high_score_key], example[low_score_key]]

        # ref_score = -1.0 if not in use, remove it if that's the case
        if scores[0] < 0:
            sentences = sentences[1:]
            scores = scores[1:]

        # Finding the indexes for the highest and lowest scores
        highest_score_index = scores.index(max(scores))
        lowest_score_index = scores.index(min(scores))

        # Assigning the corresponding sentences
        highest_score_sentence = f"{sentences[highest_score_index]}\n"
        lowest_score_sentence = f"{sentences[lowest_score_index]}\n"
        return highest_score_sentence, lowest_score_sentence

    
    def meet_requirements(prompt_tok: list[float], example: dict[str, str], target_lang: str) -> bool:
        """Check if the current sample meets certain criteria"""
        # if prompt is too long
        if len(prompt_tok) > args.max_source_length:
            #print(f"{example}\ndiscarded due to length {len(prompt_tok)} > {args.max_source_length}\n")
            return False

        # if the order is fixed, e.g., it has to be en->de
        if "required_directions" in example and example["required_directions"] != "":
            tgt = example["required_directions"].split("-")[1]
            if tgt != target_lang:
                #print(f"{example}\ndiscarded due to target language {tgt} != {target_lang}\n")
                return False

        # If using X-alma preference dataset
        if "chosen" in example:
            return True
        
        # Assert that the rejected translation is not the same as the chosen
        # use fuzzy equality to account for floating point errors
        high_score_key = f"high_{target_lang}_{args.cpo_scorer}"
        low_score_key = f"low_{target_lang}_{args.cpo_scorer}"
        ref_score_key = f"ref_{target_lang}_{args.cpo_scorer}"
        scores = [example[ref_score_key], example[high_score_key], example[low_score_key]]
        scores.sort(reverse=True)
        hi = scores[0]
        lo = scores[-1] if scores[-1] >= 0 else scores[-2]
        if isclose(hi, lo, rel_tol=0.0, abs_tol=1e-8):
            #print(f"{example}\ndiscarded due being too similar {hi} ~ {lo}\n")
            return False
            
        return True 

    
    def cpo_prompt_function(examples: list[dict[str, str]]) -> dict[str, list[str]]:
    #def cpo_prompt_function(ex: dict[str, str]) -> dict[str, list[str]]:
        """Apply the correct prompt format to each example and return a dict of the same examples with prompt template applied"""
        new_examples = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }

        for ex in examples["translation"]:
            source_lang, target_lang = ex["language_pair"].split("-") 
            if f"{source_lang}-{target_lang}" in pairs:
                prompt = get_prompt(source_lang, target_lang, ex, args.prompt_style)
                #prompt_tok = tokenizer(prompt, max_length = args.max_source_length, padding = True, truncation = True, add_special_tokens = True).input_ids
                prompt_tok = tokenizer(prompt, add_special_tokens = True).input_ids
                
                if meet_requirements(prompt_tok, ex, target_lang):
                    new_examples["prompt"].append(prompt)
                    chosen, rejected = get_chosen_reject(ex, target_lang)
                    new_examples["chosen"].append(chosen)
                    new_examples["rejected"].append(rejected)

            # No need to do reversed order when I specify all six directions
            """
            # Reversed order
            if f"{target_lang}-{source_lang}" in pairs:
                prompt = get_prompt(target_lang, source_lang, ex, args.prompt_style)
                #prompt_tok = tokenizer(prompt, max_length = args.max_source_length, padding = True, truncation = True, add_special_tokens = True).input_ids
                prompt_tok = tokenizer(prompt, add_special_tokens = True).input_ids
                
                if meet_requirements(prompt_tok, ex, source_lang):
                    new_examples["prompt"].append(prompt)
                    chosen, rejected = get_chosen_reject(ex, source_lang)
                    new_examples["chosen"].append(chosen)
                    new_examples["rejected"].append(rejected)
            """
            
        return new_examples


    # Preprocessing the datasets
    train_datasets, eval_datasets, test_datasets = None, None, None
    processed_datasets = []
    #if args.cpo_data_path:
    # dict[key: str, data: dict[str, str]]
    for lg_pair, raw_data in train_raw_data["mmt"].items():
        train_dataset = raw_data["train"]
        
        if args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        train_dataset = train_dataset.map(
            cpo_prompt_function,
            batched=True,
            batch_size=1,
            #num_proc=args.preprocessing_num_workers,
            remove_columns=["translation"],
            #remove_columns=train_dataset.column_names,
            desc = "Running CPO preprocessing",
        )

        processed_datasets.append(train_dataset)
    
    train_datasets = concatenate_datasets(processed_datasets)
    train_datasets = train_datasets.shuffle(seed=args.seed)

    return train_datasets




