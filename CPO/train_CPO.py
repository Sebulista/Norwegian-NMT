"""Program to train models with preference optimization either CPO or ARPO
Remember to set paths """

import os

#Set TOKENIZERS_PARALLELISM so it doesn't annoy us
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import datasets
from datasets import load_dataset, load_from_disk
import torch
import wandb
import random

from tqdm import tqdm
import numpy as np
import sys
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed
from trl import CPOTrainer, CPOConfig
import peft
from peft import LoraConfig, get_peft_model, PeftModel
from argparse import ArgumentParser
from typing import Literal
from pathlib import Path
import regex as re

from sacrebleu.metrics import BLEU, CHRF
from comet import download_model, load_from_checkpoint

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

from utils import preprocess_cpo_data, alma_cpo_config, alma_cpo_trainer

MODELS = {
    "normistral_7b" : "norallm/normistral-7b-warm",
    "normistral_11b": "norallm/normistral-11b-warm",
}

DATASETS_EN_NB = {
    "QE_epsilon": "./output/CPO_QE_normistral_en_nb_T1.0_n8_b1_sampling_e0.02.jsonl",
    "prefer_ref": "./output/CPO_normistral_en_nb_T1.0_n1_b1.jsonl",
    "alma_pref": "./data/xalma_en_nb_preference.jsonl",
    "fairseq": "./output/translation_CPO_QE__fairseq_mono_en_nb_T1.2_b16_sampling_e0.02.jsonl",
}

DATASETS_EN_NN = {
    "QE_epsilon": "./output/CPO_QE_normistral_en_nn_T1.0_n8_b1_sampling_e0.02.jsonl",
    "QE_beam" : "./output/translation_CPO_QE_normistral_en_nn_T1.0_b8.jsonl",
}

#Return language agnostic dataset names, format later DATASETS[key].format(slang, tlang)
DATASETS = {
    "QE_epsilon": "./output/CPO_QE_normistral_{}_{}_T1.0_n8_b1_sampling_e0.02.jsonl",
    "QE_beam" : "./output/translation_CPO_QE_normistral_{}_{}_T1.0_b8.jsonl",
    "prefer_ref": "./output/CPO_normistral_{}_{}_T1.0_n1_b1.jsonl",
    "MBR_chrf": "./output/CPO_MBR_chrf_pp_normistral_{}_{}_T1.0_n8_b1_sampling_e0.02.jsonl",
    "MBR_comet": "./output/CPO_MBR_comet_normistral_{}_{}_T1.0_n8_b1_sampling_e0.02.jsonl",
    "MBR_bleu": "./output/CPO_MBR_bleu_normistral_{}_{}_T1.0_n8_b1_sampling_e0.02.jsonl",
}

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--id", type = str, required = True, help = "unique id")
    parser.add_argument('--model', type=str, choices = MODELS.keys(), required = True, help= f"The model to use, available models: {','.join(MODELS.keys())}")
    parser.add_argument('--dataset', type=str, required = True, choices = set(DATASETS_EN_NB.keys()) | set(DATASETS.keys()), help= f"The dataset to use")
    parser.add_argument('--lora_adapter', type = str, default = None, help = "Path to already trained lora adapter. Will be ignored if not provided")
    parser.add_argument('--train', type=str, choices = ["25k", "50k"], default = "25k", help = "Which datasets to use for training")
    parser.add_argument('--val', type=str, help = "Path to validation set")
    parser.add_argument('--save', action = "store_true", help = "Whether or not to save the model")
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size')
    parser.add_argument('--epochs', type=int, default=5, help='The number of epochs to train')
    parser.add_argument('--lr', type=float, default=2e-5, help='The learning rate')
    parser.add_argument('--ga', type=int, default = 1, help = "Gradient accumulation steps, default = 1")
    parser.add_argument('--seed', type=int, default=42, help='The random seed')
    parser.add_argument('--warmup_steps', type=float, default=0.02, help='The number of warmup steps given in percent of all training steps')
    parser.add_argument('--gradient_clipping', type=float, default=10.0, help='The gradient clipping value')
    parser.add_argument('--patience', "-p", type=int, default = 3, help = "The number of iterations allowed with decreased validation loss before stopping early")
    parser.add_argument('--dropout', type = float, default = 0.1, help = "Amount of dropout")
    parser.add_argument('--scheduler', type=str, default = "cosine", choices = ["linear", "cosine", "constant", "constant_with_warmup"], help = "Which learning rate scheduler to use")
    parser.add_argument("--optim", type = str, default = "adamw_torch", choices = ["adamw_hf", "adamw_torch", "adamw_torch_fused", "adamw_apex_fused", "adamw_anyprecision", "adafactor", "adamw_bnb_8bit", "rmsprop"])
    parser.add_argument('--wandb_project', type=str, default=None, help = "IF wandb project is specified, wandb logging is activated.")
    parser.add_argument('--early_stopping', type=str, default=None, choices = ["metric", "loss"], help = "Criterion for early stopping: 'metric' or 'loss'. If not provided, early stopping is disabled.")

    parser.add_argument("--slang", type = str, choices = ["en", "nn", "nb"])
    parser.add_argument("--tlang", type = str, choices = ["en", "nn", "nb"])
    
    mixed_precision = parser.add_mutually_exclusive_group()
    mixed_precision.add_argument("--fp16", action="store_true", help = "Use half precision for training")
    mixed_precision.add_argument("--bf16", action = "store_true", help = "Use bfloat16 mixed precision")
    parser.add_argument("--tf32", action = "store_true", help = "Enable TF32 (already enabled by default?)")
    parser.add_argument("--compile", action = "store_true", help = "Use PyTorch 2.0 compile")
    parser.add_argument("--grad_cp", action = "store_true", help = "Use Gradient checkpointing")

    parser.add_argument("--no_save", action = "store_true", help = "Whether to not save")

    #lora args
    parser.add_argument("--r", type = int, default = 32, help = "Lora rank parameter")
    parser.add_argument("--lora_alpha", type = int, default = 64, help = "Lora alpha parameter")
    parser.add_argument("--lora_dropout", type = int, default = 0.05, help = "Lora dropout parameter")
    parser.add_argument("--target_modules", type = str, nargs = '+', default = "all-linear", help = "Lora modules to target. If not set, target all linear modules")

    #data args
    parser.add_argument("--cpo_scorer", type = str, choices = ["comet", "kiwi", "MBR_chrf_pp", "MBR_comet", "MBR_bleu"], default = "kiwi", help = "Which CPO scorer to use for the chosen/rejected samples")
    parser.add_argument("--max_source_length", type = int, default = 2048-1, help = "Max length prompt, longer inputs will be disregarded.")
    parser.add_argument("--prompt_style", type = str, default = "basic", choices = ["basic", "instruct"], help = "Which prompt style to use")
    parser.add_argument("--cpo_data_path", type = str, help = "Path to CPO preference dataset in jsonl format")
    parser.add_argument("--max_train_samples", type = int, help = "Max train samples, will be ignored if not provided")

    #cpo args
    parser.add_argument("--beta", type = float, default = 0.1, help = "Beta factor in CPO loss")
    parser.add_argument("--loss", type = str, default = "sigmoid", choices = ["sigmoid", "hinge", "ipo", "simpo"], help = "Which loss function to use")
    parser.add_argument("--cpo_alpha", type = float, default = 1.0, help = "CPO alpha parameter, set to 0.0 for simpo")

    parser.add_argument("--arpo", action = "store_true", help = "Whether or not to activate ARPO loss instead of CPO loss")
    parser.add_argument("--relax_eta", type = float, default = 0.9, help = "Relax coefficient 1 for ARPO")
    parser.add_argument("--eta", type = float, default = 0.4, help = "Relax coefficient 2 for ARPO, seems to be ETA from the paper")

    #eval args
    parser.add_argument("--beams", type = int, default = 1, help = "Number of beams for beam search. Currently only used by normistral, default = 1 (greed)")
    parser.add_argument("--logging_steps", type = int, default = 100, help = "Number of steps before logging (to wandb)")
    parser.add_argument("--cometkiwi", action = "store_true", help = "Whether or not to evaluate with cometkiwi too")
    parser.add_argument("--comet", action = "store_true", help = "Whether or not to evaluate with comet too")
    parser.add_argument("--check_repetition", action = "store_true", help = "Whether or not to check for repetitions in the hypotheses")
    
    args = parser.parse_args()

    return args

def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def init(args):
    #set random seed
    seed_everything(args.seed)

    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sys.stderr.write(f"Running on {device}.")

    tokenizer_path = MODELS[args.model]

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        padding_side = "left",
        add_eos_token = True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODELS[args.model],
    )

    print("Successfully loaded model to memory")

    # Load lora checkpoint if provided
    if args.lora_adapter is not None:
        peft_model = PeftModel.from_pretrained(model, args.lora_adapter)
        model = peft_model.merge_and_unload()
        print(f"Successfully loaded and merged LORA checkpoint {args.lora_adapter}")

    #Add pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '<pad>' if args.model == "normistral_11b" else tokenizer.eos_token
        print(f"Added pad token: {tokenizer.pad_token}, id = {tokenizer.pad_token_id}")


    return device, tokenizer, model


def init_comet():
    scorer = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da", saving_directory = "/path/to/models", local_files_only = True))
    scorer.eval()

    return scorer

def format_comet(sources: list[str], hypotheses: list[str], references: list[str]) -> list[dict[str, str]]:
    """Format all sents in a corpus"""
    data = []

    for src, hyp, ref in zip(sources, hypotheses, references):
        entry = {
            "src": src,
            "mt": hyp,
            "ref": ref,
        }
        data.append(entry)

    return data

def init_comet_kiwi():
    scorer = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da", saving_directory = "path/to/models", local_files_only = True))
    scorer.eval()

    return scorer

def format_kiwi(sources: list[str], hypotheses: list[str]) -> list[dict[str, str]]:
    data = []

    for src, hyp in zip(sources, hypotheses):
        entry = {
            "src": src,
            "mt": hyp,
        }
        data.append(entry)

    return data

LANGUAGE_IDS = {
    "en": "Engelsk",
    "nb": "Bokmål",
    "nn": "Nynorsk"
}

def format_prompt(sent: str, args, slang: str, tlang: str) -> str:
    if args.prompt_style == "basic":
        text = f"""{LANGUAGE_IDS[slang]}: {sent}
{LANGUAGE_IDS[tlang]}:"""
        
    elif args.prompt_style == "instruct":
        text = f"""Oversett dette fra {LANGUAGE_IDS[slang]} til {LANGUAGE_IDS[tlang]}: 
{LANGUAGE_IDS[slang]}: {sent}
{LANGUAGE_IDS[tlang]}:"""
    
    return text

@torch.no_grad()
def translate_batch(sentences: list[str], model, tokenizer, device, slang: Literal["nb", "en", "nn"], tlang: Literal["nb", "en", "nn"], max_new_tokens: int = 512) -> list[str]:
    max_new_tokens = min(int(max([len(t) for t in tokenizer(sentences).input_ids]) * 1.5), max_new_tokens)

    texts = [format_prompt(sent, args, slang, tlang) for sent in sentences]
    
    model_inputs = tokenizer(texts, return_tensors='pt', return_token_type_ids = False, padding = True).to(device)
    input_length = model_inputs.input_ids.shape[1]
        
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=args.beams,
        eos_token_id=None if args.model == "normistral_11b" else tokenizer('\n').input_ids
    )
    
    #return tokenizer.batch_decode(generated_ids[:, input_length:])
    # Default to splitting at new line
    return [sent.split("\n")[0].strip() for sent in tokenizer.batch_decode(generated_ids[:, input_length:])]

def get_dataset(slang: str, tlang: str) -> tuple[Path, bool]:
    dataset_path = Path("path/to/validation_sets")

    dataset = None
    for child in dataset_path.iterdir():
        if child.is_file():
            if child.name == f"val.{slang}_{tlang}" or child.name == f"val.{tlang}_{slang}":
                dataset = child
                break
                
    assert dataset is not None, f"Did not find validation set for {slang}_{tlang} or {tlang}_{slang}"
    
    # Swap direction if 'slang' is the target language of the file, and vice versa
    swap = False
    
    #Find source and target language from file extension
    s, t = dataset.name.split(".")[-1].split("_")
    #Correct way
    if s == slang and t == tlang:
        swap = False
    #Wrong way
    elif s == tlang and t == slang:
        swap = True

    return dataset, swap


def check_repetition(sents: list[str], verbose: bool = True):
    # Find repeating phrases
    pattern = re.compile(r"\b(\b(?:\w+\s+){1,}\w+\b)\s+\1\b")
    counter = 0
    for line in sents:
        if pattern.search(line):
            if verbose and counter < 10:
                print(f"{line}\n")
            counter += 1

    return counter


def evaluate(model, tokenizer, device, slang: Literal["nb", "en", "nn"], tlang: Literal["nb", "en", "nn"], comet_kiwi: bool = False, comet: bool = False, batch_size: int = 64, check_for_repetition: bool = False) -> dict[str, float]:
    bleu = BLEU()
    chrf = CHRF()
    
    test_set, swap = get_dataset(slang, tlang)
    with open(test_set, "r", encoding = "utf-8") as test_file:
        lines = test_file.read().strip().split("\n")

    all_references = []
    all_hypotheses = []
    all_sources = []
    
    for i in tqdm(range(0, len(lines), batch_size)):
        references = [line.split("\t")[0 if swap else 1].strip() for line in lines[i:i+batch_size]]
        all_references.extend(references)
        
        source_sents = [line.split("\t")[1 if swap else 0].strip() for line in lines[i:i+batch_size]]

        # Also accumulate source sentences for comet scoring
        if comet_kiwi or comet:
            all_sources.extend(source_sents)
            
        try:
            hypotheses = translate_batch(source_sents, model, tokenizer, device, slang, tlang)
            all_hypotheses.extend(hypotheses)
        except RuntimeError:
            print("RuntimeError")

    random_indices = [random.randint(0, len(all_references) - 1) for _ in range (15)]
    print("\nExample translations:")
    for idx in random_indices:
        print(f"REF: {all_references[idx]}")
        print(f"HYP: {all_hypotheses[idx]}\n")
    
    bleu_score = bleu.corpus_score(all_hypotheses, [all_references]).score
    chrf_score = chrf.corpus_score(all_hypotheses, [all_references]).score

    scores = {
        "bleu": bleu_score,
        "chrf": chrf_score,
    }

    if comet:
        scorer = init_comet()
        score_data = format_comet(all_sources, all_hypotheses, all_references)
        system_score = scorer.predict(score_data, batch_size = batch_size, gpus = 1, progress_bar = False)["system_score"]

        scores["comet"] = system_score
        
    
    if comet_kiwi:
        scorer = init_comet_kiwi()
        score_data = format_kiwi(all_sources, all_hypotheses)
        system_score = scorer.predict(score_data, batch_size = batch_size, gpus = 1, progress_bar = False)["system_score"]

        # Append to scores dict
        scores["cometkiwi"] = system_score

    
    if check_for_repetition:
        n = check_repetition(all_hypotheses)
        scores["repetitions"] = n
        print(f"Found {n} lines with repetitions.")

    return scores


def get_gpu_vram_info():
    if torch.cuda.is_available():
        # Get number of GPUs
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            # Get the name of the GPU
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            # Get the total VRAM for the GPU
            vram_in_bytes = torch.cuda.get_device_properties(i).total_memory
            vram_in_gb = vram_in_bytes / (1024 ** 3)
            print(f"Total VRAM: {vram_in_gb:.2f} GB")
    else:
        print("No GPU detected.")

if __name__ == "__main__":
    get_gpu_vram_info()
    args = parse_arguments()
    random.seed(args.seed)


    """if args.slang in ["en", "nn"] and args.tlang in ["en", "nn"]:
        args.cpo_data_path = DATASETS_EN_NN[args.dataset]
        print(f"Loaded ENGLISH-NYNORSK dataset: {args.cpo_data_path}")
    elif args.slang in ["en", "nb"] and args.tlang in ["en", "nb"]:
        args.cpo_data_path = DATASETS_EN_NB[args.dataset]
        print(f"Loaded ENGLISH-BOKMÅL dataset: {args.cpo_data_path}")"""

    device, tokenizer, model = init(args)

    if args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    #pairs = set(args.language_pairs.split(","))
    train_raw_data, valid_raw_data, test_raw_data = {}, None, None
    # load cpo dataset
    train_raw_data["mmt"] = {}
    
    if args.slang is None or args.tlang is None:
        # Train for all six directions
        pairs = ["en-nb", "en-nn", "nb-en", "nb-nn", "nn-nb", "nn-en"]
    else:
        pairs = [f"{args.slang}-{args.tlang}"]

    #src_lang = args.slang
    #tgt_lang = args.tlang
    #lang_string = f"{src_lang}-{tgt_lang}"
    
    for pair in pairs:
        src_lang, tgt_lang = pair.split("-")

        # EN-NB specific dataset
        if args.dataset not in DATASETS.keys():
            dataset = DATASETS_EN_NB[args.dataset]
        else:
            dataset = DATASETS[args.dataset].format(src_lang, tgt_lang)
        
        train_raw_data["mmt"][pair] = load_dataset(
            "json",
            data_files = dataset,
        )

    
    train_datasets = preprocess_cpo_data(train_raw_data, pairs, tokenizer, args)

    
    lora_config = LoraConfig(
    	r = args.r, #32
    	lora_alpha = args.lora_alpha, #64
    	lora_dropout = args.lora_dropout, #0.05
    	#target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    	target_modules = args.target_modules, #"all-linear"
    	task_type = "CAUSAL_LM",
    )

    if args.arpo:
        cfg = alma_cpo_config.CPOConfig
    else:
        cfg = CPOConfig
    
    training_args = cfg(
        f"/SET_OUTPUT_PATH/{'ARPO' if args.arpo else 'CPO'}_{args.model}_{args.id}",
        evaluation_strategy="no",
        save_strategy="no" if args.no_save else "epoch",
        learning_rate=args.lr,
        gradient_accumulation_steps = args.ga,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=args.epochs,
        num_train_epochs=args.epochs,
        #predict_with_generate=True,
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.tf32,
        optim=args.optim,
        lr_scheduler_type = args.scheduler,
        #label_smoothing_factor = 0.1,
        max_grad_norm = 1.0,
        warmup_ratio = 0.01,
        dataloader_pin_memory = True,
        torch_compile = args.compile,
        gradient_checkpointing = args.grad_cp,
        report_to="wandb",  # enable logging to W&B
        logging_steps = args.logging_steps, 
        run_name=f"{'ARPO' if args.arpo else 'CPO'}_{args.model}_{args.lr}_{args.id}",
        max_length = 2048-1,
        max_prompt_length = 1024-1, #?
        beta = args.beta,
        loss_type = args.loss,
        cpo_alpha = args.cpo_alpha,
    )
    if args.arpo:
        training_args.relax_cofficient_1 = args.relax_eta#0.9
        training_args.relax_cofficient_2 = args.eta#0.4
        

    if args.arpo:
        trnr = alma_cpo_trainer.CPOTrainer
    else:
        trnr = CPOTrainer
    
    trainer = trnr(
        model,
        args = training_args,
        train_dataset = train_datasets,
        #eval_dataset = eval_datasets,
        tokenizer = tokenizer,
        peft_config = lora_config,
    )

    trainer.train()

    # Evaluate
    for pair in pairs:
        src_lang, tgt_lang = pair.split("-")
        scores = evaluate(model, tokenizer, device, src_lang, tgt_lang, comet_kiwi = args.cometkiwi, comet = args.comet, check_for_repetition=args.check_repetition)
        mono_bleu = scores["bleu"]
        mono_chrf = scores["chrf"]
        #nb_en_bleu, nb_en_chrf = evaluate(model, tokenizer, device, "en", "nb")
    
        print(f"{mono_bleu=}, {mono_chrf=}")
        #print(f"{nb_en_bleu=}, {nb_en_chrf=}")
        
        #Log evaluation metric
        if wandb.run is not None:
            wandb.log({f"validate/{src_lang}_{tgt_lang}_BLEU": mono_bleu})
            wandb.log({f"validate/{src_lang}_{tgt_lang}_CHRF": mono_chrf})
    
            if args.cometkiwi:
                print(f"CometKiwi= {scores['cometkiwi']}")
                wandb.log({f"validate/{src_lang}_{tgt_lang}_CometKiwi": scores["cometkiwi"]})
    
            if args.comet:
                print(f"Comet= {scores['comet']}")
                wandb.log({f"validate/{src_lang}_{tgt_lang}_Comet": scores["comet"]})
    
            if args.check_repetition:
                wandb.log({"validate/repetitions": scores["repetitions"]})
        
