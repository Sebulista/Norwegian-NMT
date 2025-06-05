""" Code for generating translations for each fairseq model checkpoint on the validation set """
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Set TOKENIZERS_PARALLELISM so it doesn't annoy us
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable CUDA Launch Blocking to help debug CUDA errors
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.generation import LogitsProcessor
from peft import PeftModel

import sentencepiece as spm
from fairseq.models.transformer import TransformerModel
from fairseq.data.data_utils import collate_tokens
from fairseq import checkpoint_utils, options, tasks, utils
import sys


from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import random

from mbrs.metrics import MetricBLEU, MetricChrF, MetricCOMET, MetricCOMETkiwi, MetricMetricX, MetricTER, MetricXCOMET
from mbrs.decoders import DecoderMBR, DecoderEnsembleMBR


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--slang", type = str, choices = ["nb", "nn", "en"], help = "source language")
    parser.add_argument("--tlang", type = str, choices = ["nb", "nn", "en"], help = "target language")
    parser.add_argument("--method", type = str, help = "Translation method (model)", choices = ["fairseq_mono", "fairse_multi"])
    parser.add_argument("--batch_size", type = int, default = 1, help = "Batch size")
    parser.add_argument("--output_dir", type = str, default = None, help = "Name of output directory")
    parser.add_argument("--output_suffix", type = str, default = "", help = "Additional suffix to the output directory")
    parser.add_argument("--seed", type = int, default = 73, help = "Random seed")
    parser.add_argument("--beams", type = int, default = 5, help = "Number of beams for beam search.")
    #parser.add_argument("--dataset", type = str, default = "test_3500.en_nb")
    #parser.add_argument("--output", type = str, default = "hypotheses")

    #NorMistral args
    parser.add_argument("--n_shots", type = int, default = None)
    parser.add_argument("--normistral", type = str, choices = ["7", "11"], help = "Whether to use normistral 7b or 11b")
    parser.add_argument("--prompt_style", type = str, choices = ["basic", "instruct"], default = "basic", help = "Which prompt formatting style to use. Basic = zero-shot, instruct = zero-shot with initial instruction line")
    parser.add_argument("--lora_merge", type = str, default = None, help = "Path to lora checkpoint to merge with main model. Won't be used if not provided")
    parser.add_argument("--lora", type = str, default = None, help = "Path to lora checkpoint. Won't be used if not provided.")
    parser.add_argument("--max_new_tokens", type = int, default = 256, help = "Max new number of tokens to generate for normistral, default = 256")
    parser.add_argument("--bf16", action = "store_true", help = "If provided, load normistral in bfloat16 instead of float16")

    #Fairseq args
    parser.add_argument("--spm", type=str, default = "", help = "Path to sentencepiece model")
    parser.add_argument("--model_dir", type=str, default = "", help = "Path to model dir with dict")
    parser.add_argument("--checkpoint_dir", type = str, default = "", help = "Path to dir with checkpoints")
    parser.add_argument("--model", type=str, default = "", help = "Path to fairseq checkpoint")

    #fairseq multi args
    parser.add_argument("--encoder_langtok", type = str, choices = ["tgt", "src"], default = None, help = "Whether to use source side or target side encoder langtoken")
    parser.add_argument("--decoder_langtok", action = "store_true", help = "Whether to add a decoder langtoken")
    parser.add_argument("--tagged_bt", action = "store_true", help = "Whether to add tagged backtranslation tokens for compatability.")


    args = parser.parse_args()
    
    return args

class AbstractTranslator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"SYSTEM: Running on {self.device}", flush=True)

    def translate(self, sentence: str) -> str:
        raise NotImplementedError


class FairseqMonoTranslator(AbstractTranslator):
    def __init__(self, args):
        super().__init__(args)

        model_dir = Path(args.model_dir)
        model_path = model_dir / args.model
        spm_model = model_dir / args.spm

        
        # Initialize the Fairseq parser and add arguments as done in fairseq-interactive
        parser = options.get_interactive_generation_parser()

        self.fairseq_args = options.parse_args_and_arch(parser, input_args=[
            str(model_dir),
            '--path', str(model_path),
            '--task', 'translation',
            '--source-lang', args.slang,
            '--target-lang', args.tlang,
            '--bpe', 'sentencepiece',
            '--sentencepiece-model', str(spm_model),
            '--sacrebleu',
            '--remove-bpe', 'sentencepiece',
            '--beam', str(args.beams),
            '--user-dir', '../custom_fairseq'
        ])
    
        
        # Load the task
        self.task = tasks.setup_task(self.fairseq_args)

        self.models, _model_args = utils.load_ensemble_for_inference([str(model_path)], self.task)
        
        for model in self.models:
            model.to(self.device).eval()
        
        # Initialize the BPE process
        self.bpe = self.task.build_bpe(self.fairseq_args)
        
        # Build generator
        self.generator = self.task.build_generator(self.models, self.fairseq_args)


    @torch.no_grad()
    def translate(self, sentence: str) -> str:
        # Tokenize input text using SentencePiece
        tokenized_input = self.bpe.encode(sentence)
    
        # Convert to tensor and predict
        input_ids = self.task.src_dict.encode_line(tokenized_input, append_eos=True).long().unsqueeze(0).to(self.device)
        lengths = torch.LongTensor([input_ids.size(1)]).to(self.device)
        
        sample = {'net_input': {'src_tokens': input_ids, 'src_lengths': lengths}}
    
        # Perform the translation
        translations = self.task.inference_step(self.generator, self.models, sample)


        translation_tokens = translations[0][0]['tokens']
        
        # Convert tokens to text
        decoded_output = self.task.tgt_dict.string(translation_tokens)
        
        # Detokenize output using SentencePiece
        detokenized_output = self.bpe.decode(decoded_output)
        
        return detokenized_output

    @torch.no_grad()
    def translate_batch(self, sentences: list[str]) -> list[str]:
        # Tokenizing input sentences and creating batch
        tokens_list = [
            self.task.src_dict.encode_line(
                line=self.bpe.encode(sentence),
                add_if_not_exist=False,
                append_eos=True
            ).long()
            for sentence in sentences
        ]
        
        src_lengths = torch.LongTensor([tokens.size(0) for tokens in tokens_list]).to(self.device)
        src_tokens = collate_tokens(
            tokens_list, pad_idx=self.task.src_dict.pad(), eos_idx=self.task.src_dict.eos()
        ).to(self.device)
        
    
        sample = {
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
        }
    
        # Make inference
        translations = self.task.inference_step(self.generator, self.models, sample)
    
        # Decode the output for each sample in the batch
        results = []
        for i, translation in enumerate(translations):
            output_tokens = translation[0]['tokens']
            
            decoded_output = self.bpe.decode(self.task.tgt_dict.string(output_tokens))
    
            results.append(decoded_output)
    
        return results

class FairseqMultiTranslator(AbstractTranslator):
    def __init__(self, args):
        super().__init__(args)
        
        model_dir = Path(args.model_dir)
        model_path = args.model
        spm_model = model_dir / args.spm

        input_args=[
            str(model_dir),
            '--user-dir', '../custom_fairseq',
            '--path', str(model_path),
            '--task', 'translation_multi_simple_epoch',
            '--source-lang', args.slang,
            '--target-lang', args.tlang,
            '--bpe', 'sentencepiece',
            '--sentencepiece-model', str(spm_model),
            '--sacrebleu',
            '--remove-bpe', 'sentencepiece',
            '--beam', str(args.beams),
            '--lang-pairs', "nb-nn,nn-nb"
        ]

        if args.encoder_langtok is not None:
            input_args.extend([
                '--encoder-langtok', args.encoder_langtok,
            ])

        if args.decoder_langtok:
            input_args.extend([
                '--decoder-langtok',
            ])

        if args.tagged_bt:
            input_args.extend([
                '--langtoks-specs', 'main,dae',
                '--langtoks', '{"main": (None, "tgt"), "bt": ("src.dae", "tgt")}',
            ])
        
        # Initialize the Fairseq parser and add arguments as done in fairseq-interactive
        parser = options.get_interactive_generation_parser()
        self.fairseq_args = options.parse_args_and_arch(parser, input_args = input_args)
        

        self.source_langtoken = f"__{args.slang}__" if args.encoder_langtok == "src" else f"__{args.tlang}__"
        self.langtoken = f"__{args.tlang}__"
        
        # Load the task
        self.task = tasks.setup_task(self.fairseq_args)

        self.models, _model_args = utils.load_ensemble_for_inference([str(model_path)], self.task)
        
        self.use_cuda = torch.cuda.is_available()
        
        for model in self.models:
            model.eval()
            if self.use_cuda:
                model.cuda()
        
        # Initialize the BPE process
        self.bpe = self.task.build_bpe(self.fairseq_args)
        
        # Build generator
        self.generator = self.task.build_generator(self.models, self.fairseq_args)

    def _preprocess_sentence(self, sentence: str) -> str:
        # This is done automatically in the infernence step or generator of the task
        #sentence = f"{self.source_langtoken} {sentence}"
        return self.bpe.encode(sentence)
    
    @torch.no_grad()
    def translate(self, sentence: str) -> str:
        # Tokenize input text using SentencePiece
        tokenized_input = self._preprocess_sentence(sentence)
    
        # Convert to tensor and predict
        input_ids = self.task.source_dictionary.encode_line(tokenized_input, append_eos=True).long().unsqueeze(0)
        lengths = torch.LongTensor([input_ids.size(1)])
    
        if self.use_cuda:
            input_ids = input_ids.cuda()
            lengths = lengths.cuda()
        
        sample = {'net_input': {'src_tokens': input_ids, 'src_lengths': lengths}}
    
        # Perform the translation
        translations = self.task.inference_step(self.generator, self.models, sample)
        translation_tokens = translations[0][0]['tokens']
        
        # Convert tokens to text
        decoded_output = self.task.target_dictionary.string(translation_tokens)
    
        # Detokenize output using SentencePiece
        detokenized_output = self.bpe.decode(decoded_output)
    
        return detokenized_output

    @torch.no_grad()
    def translate_batch(self, sentences: list[str]) -> list[str]:
        # Tokenizing input sentences and creating batch
        tokens_list = [
            self.task.source_dictionary.encode_line(
                line=self._preprocess_sentence(sentence),
                add_if_not_exist=False,
                append_eos=True
            ).long()
            for sentence in sentences
        ]
        
        src_lengths = torch.LongTensor([tokens.size(0) for tokens in tokens_list])
        src_tokens = collate_tokens(
            tokens_list, pad_idx=self.task.source_dictionary.pad(), eos_idx=self.task.source_dictionary.eos()
        )
        
        if self.use_cuda:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()
    
        sample = {
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
        }
    
        # Make inference
        translations = self.task.inference_step(self.generator, self.models, sample)
    
        # Decode the output for each sample in the batch
        results = []
        for i, translation in enumerate(translations):
            output_tokens = translation[0]['tokens']
            decoded_output = self.task.target_dictionary.string(output_tokens)
    
            decoded_output = self.bpe.decode(decoded_output)

            if decoded_output.startswith(self.langtoken):
                decoded_output = decoded_output[7:]
    
            results.append(decoded_output)
    
        return results

def get_datasets(slang: str, tlang: str):
    """Get all relevant validation sets from the specified source path"""
    source_path = Path("path/to/validation_sets")
    datasets = []
    #Swap direction if 'slang' is the target language of the file, and vice versa
    swap = False
    
    for file in source_path.iterdir():
        if file.name.startswith("valid_raw"):
            #Find source and target language from file extension
            s, t = file.name.split(".")[-1].split("_")
            #Correct way
            if s == slang and t == tlang:
                datasets.append(file)
                swap = True
            #Wrong way
            elif s == tlang and t == slang:
                datasets.append(file)
                swap = False

    #Return list of datasets with a boolean that tells if the dataset is the right direction
    return datasets, swap

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
    random.seed(73)
    get_gpu_vram_info()
    print(f"\n{' '.join(sys.argv)}\n")
    args = parse_args()

    #Get all checkpoints
    checkpoints = []
    for child in Path(args.checkpoint_dir).iterdir():
        #if child.name not in ["checkpoint_best.pt", "checkpoint_last.pt"]:
        if child.name.endswith("pt"):
            checkpoints.append(child)
    #print(checkpoints, flush = True)
    

    if args.output_dir is None:
        output_dir = Path(".") / f"{args.checkpoint_dir.split('/')[-1].strip()}__{args.slang}_{args.tlang}{args.output_suffix}"
    else:
        output_dir = Path(".") / args.output_dir
    output_dir.mkdir()

    for checkpoint in checkpoints:
        print(checkpoint)
        
        args.model = checkpoint
        if args.method == "fairseq_multi":
            translator = FairseqMultiTranslator(args)
        else:
            translator = FairseqMonoTranslator(args)
            
        datasets, swap = get_datasets(args.slang, args.tlang)
        dataset = datasets[0]
        
        with open(dataset, "r", encoding = "utf-8") as test_file:
            lines = test_file.read().strip().split("\n")

        output_file = output_dir / f"{checkpoint.name.split('.')[0]}.hyp"
        ref_file = output_dir / f"validation.ref"

        
        # Write references to file
        with open(ref_file, "w", encoding = "utf-8") as ref:
            for line in lines:
                reference = line.split("\t")[0 if swap else 1].strip()
                ref.write(f"{reference}\n")
        
        # All models with batch size 1
        if args.batch_size == 1:
            with open(output_file, "w", encoding = "utf-8") as hyp:
                for line in tqdm(lines):
                    reference = line.split("\t")[0 if swap else 1].strip()
                    source_sent = line.split("\t")[1 if swap else 0].strip()
                    
                    hypothesis = translator.translate(source_sent)
                    
                    hyp.write(f"{hypothesis}\n")
    
    
        else:
            with open(output_file, "w", encoding = "utf-8") as hyp:
                for i in tqdm(range(0, len(lines), args.batch_size)):
                    references = [line.split("\t")[0 if swap else 1].strip() for line in lines[i:i+args.batch_size]]
                    source_sents = [line.split("\t")[1 if swap else 0].strip() for line in lines[i:i+args.batch_size]]
    
                    hypotheses = translator.translate_batch(source_sents)
    
                    for h in hypotheses:
                        hyp.write(f"{h}\n")
        
    
