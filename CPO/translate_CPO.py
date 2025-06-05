""" Program to generate triplet preference data
Remember to set paths """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Enable CUDA Launch Blocking to help debug CUDA errors
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

#Set TOKENIZERS_PARALLELISM so it doesn't annoy us
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.generation import LogitsProcessor
from peft import PeftModel

import sentencepiece as spm
from fairseq.models.transformer import TransformerModel
from fairseq.data.data_utils import collate_tokens
from fairseq import checkpoint_utils, options, tasks, utils
from comet import download_model, load_from_checkpoint
import sys
import json
from contextlib import nullcontext
from typing import Literal
import regex as re

from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import random

from mbrs.metrics import MetricBLEU, MetricChrF, MetricCOMET
from mbrs.decoders import DecoderMBR

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

import warnings
# Ignore all warnings
#warnings.filterwarnings("ignore")
# Ignore specific warnings
warnings.filterwarnings("ignore", message="The `srun` command is available on your system but is not used.")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--slang", type = str, choices = ["nb", "nn", "en"], help = "source language")
    parser.add_argument("--tlang", type = str, choices = ["nb", "nn", "en"], help = "target language")
    parser.add_argument("--method", type = str, help = "Translation method (model)")
    parser.add_argument("--batch_size", type = int, default = 1, help = "Batch size")
    parser.add_argument("--output_dir", type = str, default = None, help = "Name of output directory")
    parser.add_argument("--seed", type = int, default = 73, help = "Random seed")
    parser.add_argument("--debug", action = "store_true", help = "Whether or not to provide a full debug file")
    #parser.add_argument("--beams", type = int, default = 10, help = "Number of beams for beam search. Currently only used by normistral, default = 10")
    parser.add_argument("--dataset", type = str, choices = ["25k", "50k"], default = "25k", help = "Which dataset split to use")
    parser.add_argument("--max_samples", type = int, default = None, help = "Number of samples to translate")
    
    parser.add_argument("--cache_dir", type = str, help = "cache directory for pretrained models")
    parser.add_argument("--scorer_model", type = str, default = "Unbabel/wmt22-comet-da", help = "Which model to use for scoring")

    #NorMistral args
    parser.add_argument("--n_shots", type = int, default = None)
    parser.add_argument("--normistral", type = str, choices = ["7", "11"], help = "Whether to use normistral 7b or 11b")
    parser.add_argument("--prompt_style", type = str, choices = ["basic", "instruct"], default = "basic", help = "Which prompt formatting style to use. Basic = zero-shot, instruct = zero-shot with initial instruction line")
    parser.add_argument("--lora", type = str, default = None, help = "Path to lora checkpoint. Won't be used if not provided.")
    parser.add_argument("--max_new_tokens", type = int, default = 256, help = "Max new number of tokens to generate for normistral, default = 256")
    parser.add_argument("--bf16", action = "store_true", help = "If provided, load normistral in bfloat16 instead of float16")
    
    #Fairseq args
    parser.add_argument("--spm", type=str, default = "", help = "Path to sentencepiece model")
    parser.add_argument("--model_dir", type=str, default = "", help = "Path to model dir with dict")
    parser.add_argument("--model", type=str, default = "", help = "Path to fairseq checkpoint")

    #fairseq multi args
    parser.add_argument("--encoder_langtok", type = str, choices = ["tgt", "src"], help = "Whether to use source side or target side encoder langtoken")

    #Decoding args
    parser.add_argument("--sampling", action = "store_true", help = "Decode using sampling")
    parser.add_argument("--top_k", type = int, default = -1, help = "k value for top k sampling") #common value: 50
    parser.add_argument("--top_p", type = float, default = -1.0, help = "p value for top p sampling") #common value: 0.9
    parser.add_argument("--min_p", type = float, default = None, help = "min_p value") #common 0.05
    parser.add_argument("--epsilon", type = float, default = -1.0, help = "epsilon cutoff for epsilon sampling") #common value 0.02

    parser.add_argument("--diverse_beam", action = "store_true", help = "Decode using diverse beam search")
    parser.add_argument("--diverse_beam_groups", type = int, default = 4, help = "Number of groups for diverse beam search")
    parser.add_argument("--diverse_beam_strength", type = float, default = 0.5, help = "Strength of diversity penalty for diverse beam search")
    
    parser.add_argument("--temp", type = float, default = 1.0, help = "Probability temperature")
    parser.add_argument("--mbr", action = "store_true", help = "If provided, use minimum bayes risk decoding")
    parser.add_argument("--n_hyp", type = int, default = 5, help = "Number of sentences to return")
    parser.add_argument("--beams", type = int, default = None, help = "Number of beams, if not provided, n_hyp will be used. Does nothing for fairseq")

    parser.add_argument("--xalma", action = "store_true", help = "Generate preference pairs xalma style. Always prefer ref")

    args = parser.parse_args()

    args.qe = "kiwi" in args.scorer_model

    # Generate only 1 translation with xalma style
    if args.xalma:
        args.n_hyp = 1

    if args.method == "fairseq_mono":
        pat = re.compile(r"(nb|en|nn)_(nb|en|nn)")
        match = re.search(pat, args.model_dir.split("/")[-1])
        if match is not None:
            src, tgt = match.group().split("_")
            args.slang = src
            args.tlang = tgt
            print(f"Inferred source and target language: {args.slang=}, {args.tlang=}")
        elif match is None and not (args.slang and args.tlang):
            raise ValueError(f"Could not infer source and target language from model path {args.model_dir}. Please provide args.slang and args.tlang")
    
    return args

class AbstractTranslator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"(COMET) GPU == {'1' if torch.cuda.is_available() else '0'}")

        self.scorer = load_from_checkpoint(download_model(args.scorer_model, saving_directory = args.cache_dir, local_files_only = True))
        self.scorer.eval()

        if args.qe:
            print("Evaluating models with quality estimation, reference score is included")
        self.format_score = self.format_comet_kiwi if args.qe else self.format_comet
        
        print(f"SYSTEM: Running on {self.device}", flush=True)

    def format_comet(self, source: str, hypotheses: list[str], reference: str) -> list[dict[str, str]]:
        data = []
        
        for hyp in hypotheses:
            entry = {
                "src": source,
                "mt": hyp,
                "ref": reference,
            }
            data.append(entry)

        return data

    def format_comet_kiwi(self, source: str, hypotheses: list[str], reference: str) -> list[dict[str, str]]:
        data = []

        # Add reference as the last element to also calculate its score
        for hyp in hypotheses+[reference]:
            entry = {
                "src": source,
                "mt": hyp,
            }
            data.append(entry)

        return data
    
    def translate(self, sentence: str) -> str:
        raise NotImplementedError



# Normistral base class, not to be used
class NorMistralBase(AbstractTranslator):
    def __init__(self, args):
        super().__init__(args)

        self.LANGUAGE_IDS = {
            "en": "Engelsk",
            "nb": "Bokmål",
            "nn": "Nynorsk"
        }
       
        #Choose between normistral 11B and 7B
        model_path = f"norallm/normistral-{'11' if args.normistral == '11' else '7'}b-warm"
        print(f"{model_path=}", flush = True)
        

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side = "left"
        )

        #Check tokenizer pad side, sometimes it does not change the pad side during init
        print(f"Tokenizer padding side: {self.tokenizer.padding_side}")
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"
            print(f"Set tokenizer padding side to \'left\'")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        ).to(self.device).eval()

        print(f"Sucessfully loaded the model to the memory")

        # Load lora checkpoint if provided
        if args.lora is not None:
            peft_model = PeftModel.from_pretrained(self.model, args.lora)
            self.model = peft_model.merge_and_unload()
            print(f"Successfully loaded LORA checkpoint {args.lora}")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '<pad>' if args.normistral == '11' else self.tokenizer.eos_token
            print(f"Added pad token: {self.tokenizer.pad_token}, id = {self.tokenizer.pad_token_id}")


    def format_prompt(self, sentence: str) -> str:
        raise NotImplementedError

    def translate(self, sentence: str) -> str:
        raise NotImplementedError
    
    def translate_batch(self, sentence: list[str]) -> list[str]:
        raise NotImplementedError



class NorMistralTranslator(NorMistralBase):
    def __init__(self, args):
        super().__init__(args)

    def format_prompt(self, sent: str) -> str:
        if self.args.prompt_style == "basic":
            text = f"""{self.LANGUAGE_IDS[self.args.slang]}: {sent}
{self.LANGUAGE_IDS[self.args.tlang]}:"""
            
        elif self.args.prompt_style == "instruct":
            text = f"""Oversett dette fra {self.LANGUAGE_IDS[self.args.slang]} til {self.LANGUAGE_IDS[self.args.tlang]}: 
{self.LANGUAGE_IDS[self.args.slang]}: {sent}
{self.LANGUAGE_IDS[self.args.tlang]}:"""
        
        return text

    @torch.no_grad()
    def translate(self, sentence: str, ref: str) -> str:
        text = self.format_prompt(sentence)
        
        input_ids = self.tokenizer(text, return_tensors='pt').input_ids.to(self.device)
        predictions = self.model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=args.beams if args.beams is not None else args.n_hyp,
            num_return_sequences=args.n_hyp,
            #normistral 11B tokenizer stops at numbers when \n is eos_token_id
            eos_token_id=None if args.normistral == '11' else self.tokenizer('\n').input_ids
        ) #-> dim(num_return_sequences, sequence_length)

        decoded_output = self.tokenizer.batch_decode(
            predictions[:, input_ids.size(1):],
            skip_special_tokens = True,
            clean_up_tokenization_spaces=True
        ) #-> flat list dim(num_return_sequences)

        hypotheses = [sentence.strip() for sentence in decoded_output]

        score_data = self.format_score(sentence, hypotheses, ref)
        # Retunrs a list of scores
        scores = self.scorer.predict(score_data, batch_size = self.args.n_hyp + 1 if args.qe else 0, gpus = 1 if self.device == "cuda:0" else 0, progress_bar = False)["scores"]

        # [0] = best, [-1] = worst
        if args.qe:
            sorted_scores = sorted(list(zip(hypotheses, scores[:-1])), key = lambda x : x[1], reverse = True)
            ref_score = (ref, scores[-1])
        else:
            sorted_scores = sorted(list(zip(hypotheses, scores)), key = lambda x : x[1], reverse = True)
            ref_score = None
        
        #Standard beam search
        return sorted_scores, ref_score # dim = (n_hyp, 2)

    
    @torch.no_grad()
    def translate_batch(self, sentences: list[str], refs: list[str]) -> list[str]:
        max_new_tokens = min(int(max([len(t) for t in self.tokenizer(sentences).input_ids]) * 1.5), args.max_new_tokens)

        texts = [self.format_prompt(sent) for sent in sentences]
        
        model_inputs = self.tokenizer(texts, return_tensors='pt', return_token_type_ids = False, padding = True).to(self.device)
        input_length = model_inputs.input_ids.shape[1]
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=args.sampling,
            num_beams=args.beams if args.beams is not None else args.n_hyp,
            num_beam_groups=args.diverse_beam_groups if args.diverse_beam else 1,
            diversity_penalty=args.diverse_beam_strength if args.diverse_beam else 0.0, #Is this the diverse beam strength argument from the paper?
            num_return_sequences=args.n_hyp,
            eos_token_id=None if args.normistral == '11' else self.tokenizer('\n').input_ids,
            temperature = args.temp if args.sampling else 1.0,
            top_k = args.top_k if (args.sampling and args.top_k > 0) else 50,
            top_p = args.top_p if (args.sampling and args.top_p > 0.0) else 1.0,
            min_p = args.min_p if args.sampling else None,
            epsilon_cutoff = args.epsilon if (args.sampling and args.epsilon > 0.0) else 0.0,
        ) #dim = (batch_size*num_return_sequences, sequence_length)

        # Aggregated comet scores dim = (batch_size * num_return_sequences, dict[str, str])
        agg_score_data = []
        # Aggregated hypotheses dim = (batch_size, num_return_sequences, str)
        agg_hypotheses = []
        # All reference scores dim = (batch_size, tuple(str, int)) if in use
        all_refs = []

        decoded_output = self.tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=True) # -> flat list dim(bsz*num_return_sequences)
        # len(texts) = batch size, args.n_hyp = num_return_sequences
        translations = (decoded_output[i*args.n_hyp : (i+1)*args.n_hyp] for i in range(len(texts))) # -> generator of dim (bsz, num_return_sequences)
        
        all_hypotheses_sorted = [] # dim = (batch_size, num_return_sequences, 2)
        for i, translation in enumerate(translations):
            hypotheses = [sentence.split("\n")[0].strip() for sentence in translation]
            agg_hypotheses.append(hypotheses)

            score_data = self.format_score(sentences[i], hypotheses, refs[i])
            agg_score_data.extend(score_data)

        # Return before scoring with xalma style
        if args.xalma:
            return agg_hypotheses, all_refs
            
        # Returns a flat list of scores no matter the batch size
        scores = self.scorer.predict(agg_score_data, batch_size = self.args.batch_size, gpus = 1 if torch.cuda.is_available() else 0, progress_bar = False)["scores"]

        # i = args.batch_size
        for i, hypotheses in enumerate(agg_hypotheses):
            # Each SORE batch of size n_hyp + 1, where the +1 is the reference score
            # This +1 is not included in the hypotheses
            if args.qe:
                sorted_scores = sorted(list(zip(hypotheses, scores[i*args.n_hyp + i : i*args.n_hyp + args.n_hyp + i])), key = lambda x : x[1], reverse = True)
                all_hypotheses_sorted.append(sorted_scores)
                all_refs.append( (refs[i], scores[i*args.n_hyp + args.n_hyp + i]) )
            else:
                # [0] = best, [-1] = worst
                sorted_scores = sorted(list(zip(hypotheses, scores[i*args.n_hyp:i*args.n_hyp+args.n_hyp])), key = lambda x : x[1], reverse = True)
                all_hypotheses_sorted.append(sorted_scores)
    
        return all_hypotheses_sorted, all_refs




class FairseqMonoTranslator(AbstractTranslator):
    def __init__(self, args):
        super().__init__(args)

        model_dir = Path(args.model_dir)
        model_path = model_dir / args.model
        spm_model = model_dir / args.spm

        print(f"Decoding strategy: {'MBR' if args.mbr else 'BEAM SEARCH'}")
        
        # Initialize the Fairseq parser and add arguments as done in fairseq-interactive
        parser = options.get_interactive_generation_parser()

        input_args = [
            str(model_dir),
            '--path', str(model_path),
            '--task', 'epsilon_translation',
            '--source-lang', args.slang,
            '--target-lang', args.tlang,
            '--bpe', 'sentencepiece',
            '--sentencepiece-model', str(spm_model),
            '--sacrebleu',
            '--remove-bpe', 'sentencepiece',
            '--beam', str(args.n_hyp),
            '--nbest', str(args.n_hyp),
            '--seed', str(args.seed),
            '--user-dir', '../custom_fairseq',
            '--temperature', str(args.temp),
        ]

        if args.diverse_beam:
            input_args += [
                '--diverse-beam-groups', str(args.diverse_beam_groups), #4
                '--diverse-beam-strength', str(args.diverse_beam_strength), #1.0
            ]

        if args.sampling:
            input_args += [
                '--sampling',
            ]

            if args.top_k > 0:
                input_args += [
                    '--sampling-topk', str(args.top_k),
                ]
            elif args.top_p > 0:
                input_args += [
                    '--sampling-topp', str(args.top_p),
                ]
            elif args.epsilon > 0:
                input_args += [
                    '--sampling-epsilon', str(args.epsilon),
                ]
        
        self.fairseq_args = options.parse_args_and_arch(parser, input_args=input_args)
        
        
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

        if args.mbr:
            # Setup COMET.
            metric_cfg = MetricCOMET.Config(
                model="Unbabel/wmt22-comet-da",
                batch_size=64,
                fp16=True,
            )
            metric = MetricCOMET(metric_cfg)

            # Setup MBR decoding.
            decoder_cfg = DecoderMBR.Config()
            self.decoder = DecoderMBR(decoder_cfg, metric)  

    
    @torch.no_grad()
    def translate(self, sentence: str, ref: str) -> str:
        # Tokenize input text using SentencePiece
        tokenized_input = self.bpe.encode(sentence)
    
        # Convert to tensor and predict
        input_ids = self.task.src_dict.encode_line(tokenized_input, append_eos=True).long().unsqueeze(0)
        lengths = torch.LongTensor([input_ids.size(1)])
    
        if self.use_cuda:
            input_ids = input_ids.cuda()
            lengths = lengths.cuda()
        
        sample = {'net_input': {'src_tokens': input_ids, 'src_lengths': lengths}}
    
        # Perform the translation
        translations = self.task.inference_step(self.generator, self.models, sample)

        # len(translations= = args.nhyp
        translation_tokens = [t['tokens'] for t in translations[0]]
        
        # Convert tokens to text
        decoded_output = [self.task.tgt_dict.string(t) for t in translation_tokens]
        # Detokenize output using SentencePiece
        hypotheses = [self.bpe.decode(d) for d in decoded_output]

        score_data = self.format_score(sentence, hypotheses, ref)
        # Retunrs a list of scores
        scores = self.scorer.predict(score_data, batch_size = self.args.n_hyp + 1 if args.qe else 0, gpus = 1 if self.device == "cuda:0" else 0, progress_bar = False)["scores"]

        # [0] = best, [-1] = worst
        if args.qe:
            sorted_scores = sorted(list(zip(hypotheses, scores[:-1])), key = lambda x : x[1], reverse = True)
            ref_score = (ref, scores[-1])
        else:
            sorted_scores = sorted(list(zip(hypotheses, scores)), key = lambda x : x[1], reverse = True)
            ref_score = None
        
        #MBR
        if args.mbr:
            mbr_decoded_output = self.decoder.decode(hypotheses, hypotheses, source=sentence, nbest=1)
            return mbr_decoded_output.sentence
        #Standard beam search
        else:
            return sorted_scores, ref_score # dim = (n_hyp, 2)

    
    @torch.no_grad()
    def translate_batch(self, sentences: list[str], refs: list[str]) -> list[str]:
        # Tokenizing input sentences and creating batch
        tokens_list = [
            self.task.src_dict.encode_line(
                line=self.bpe.encode(sentence),
                add_if_not_exist=False,
                append_eos=True
            ).long()
            for sentence in sentences
        ]
        
        src_lengths = torch.LongTensor([tokens.size(0) for tokens in tokens_list])
        src_tokens = collate_tokens(
            tokens_list, pad_idx=self.task.src_dict.pad(), eos_idx=self.task.src_dict.eos()
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
        # dim = (batch_size, n_hyp)
        translations = self.task.inference_step(self.generator, self.models, sample)
    
        # Decode the output for each sample in the batch
        results = []
        # Aggregated comet scores N=n_hyp * batch_size
        agg_score_data = []
        # Aggregated hypotheses N=[n_hyp] * batch_size
        agg_hypotheses = []
        # All reference scores N = [batch_size] if in use
        all_refs = []
        
        all_hypotheses_sorted = [] # dim = (batch_size, n_hyp, 2)
        for i, translation in enumerate(translations):
            # len(output_tokens) = args.n_hyp
            output_tokens = [t['tokens'] for t in translation]

            decoded_output = [self.task.tgt_dict.string(t) for t in output_tokens]
            hypotheses = [self.bpe.decode(d) for d in decoded_output]
            agg_hypotheses.append(hypotheses)

            score_data = self.format_score(sentences[i], hypotheses, refs[i])
            agg_score_data.extend(score_data)
            
        # Returns a flat list of scores no matter the batch size
        scores = self.scorer.predict(agg_score_data, batch_size = self.args.batch_size, gpus = 1 if torch.cuda.is_available() else 0, progress_bar = False)["scores"]

        # i = args.batch_size
        for i, hypotheses in enumerate(agg_hypotheses):
            # Each SORE batch of size n_hyp + 1, where the +1 is the reference score
            # This +1 is not included in the hypotheses
            if args.qe:
                sorted_scores = sorted(list(zip(hypotheses, scores[i*args.n_hyp + i : i*args.n_hyp + args.n_hyp + i])), key = lambda x : x[1], reverse = True)
                all_hypotheses_sorted.append(sorted_scores)
                all_refs.append( (refs[i], scores[i*args.n_hyp + args.n_hyp + i]) )
            else:
                # [0] = best, [-1] = worst
                sorted_scores = sorted(list(zip(hypotheses, scores[i*args.n_hyp:i*args.n_hyp+args.n_hyp])), key = lambda x : x[1], reverse = True)
                all_hypotheses_sorted.append(sorted_scores)
    
        return all_hypotheses_sorted, all_refs



def get_dataset(slang: str, tlang: str, split: Literal["25k", "50k"]):
    """Get all relevant test sets from the specified source path"""
    source_path = Path(f"SET_PATH/data/best_data_comet/{split}")
    
    for file in source_path.iterdir():
        #Find source and target language from file extension
        s, t = file.name.split(".")[-1].split("-")
        #Correct way
        if s == slang and t == tlang:
            return file


def format_json(args: "argparse.Namespace", source_sent: str, reference: str, sorted_hypotheses: list[tuple[str, int]], ref_score: tuple[str, int]) -> str:
    """Fill out a dictionary and return a json compatible dict string"""
    metric = "kiwi" if args.qe else "comet"

    if args.xalma:
        dict = {
            args.slang: source_sent,
            "chosen": reference,
            "reject": sorted_hypotheses[0],
            "language_pair": f"{args.slang}-{args.tlang}",
            "required_directions": f"{args.slang}-{args.tlang}",
        }
    
    else:
        dict = {
            args.slang: source_sent,
            args.tlang: reference,
            # score
            f"high_{args.tlang}_{metric}": sorted_hypotheses[0][1],
            f"low_{args.tlang}_{metric}": sorted_hypotheses[-1][1],
            # sent
            f"high_{args.tlang}": sorted_hypotheses[0][0],
            f"low_{args.tlang}": sorted_hypotheses[-1][0],
            "language_pair": f"{args.slang}-{args.tlang}",
            "required_directions": f"{args.slang}-{args.tlang}",
            f"ref_{args.tlang}_{metric}": ref_score[1] if ref_score else -1.0,
        }

    t_dict = {"translation": dict}
    
    json_string = json.dumps(t_dict, ensure_ascii = False)

    return json_string
    

def get_output_name(args: "argparse.Namespace") -> str:
    """Create a semi-unique output file name based on direction and decoding parameters"""
    decoding_string = f"n{args.n_hyp}_b{args.beams if args.beams is not None else args.n_hyp}"
    if args.diverse_beam:
        decoding_string += f"_db_{args.diverse_beam_groups}_{args.diverse_beam_strength}"
    elif args.sampling:
        decoding_string += f"_sampling"
        if args.top_k > 0:
            decoding_string += f"_k{args.top_k}"
        elif args.top_p > 0:
            decoding_string += f"_p{args.top_p}"
        elif args.epsilon > 0:
            decoding_string += f"_e{args.epsilon}"

    return f"CPO_{'QE_' if args.qe else ''}{args.method}_{args.slang}_{args.tlang}_T{args.temp}_{decoding_string}"


if __name__ == "__main__":
    random.seed(73)
    print(sys.argv)
    args = parse_args()

    if args.method == "fairseq_mono":
        translator = FairseqMonoTranslator(args)
    elif args.method == "normistral":
        translator = NorMistralTranslator(args)
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    dataset = get_dataset(args.slang, args.tlang, args.dataset)
    with open(dataset, "r", encoding = "utf-8") as test_file:
        lines = test_file.read().strip().split("\n")
        lines = lines[:args.max_samples] if args.max_samples is not None else lines
        print(f"Translating {len(lines)} sentences")
    
    output_dir = Path("./output")
    
    output_name = get_output_name(args)
    output_file = output_dir / f"{output_name}.jsonl"
    if args.debug:
        debug_file = output_dir / f"{output_name}_DEBUG.txt"
    
    if args.batch_size > 1:
        with open(output_file, "w", encoding = "utf-8") as out, \
            open(debug_file, "w", encoding = "utf-8") if args.debug else nullcontext() as debug:
            for i in tqdm(range(0, len(lines), args.batch_size)):
                references = [line.split("\t")[1].strip() for line in lines[i:i+args.batch_size]]
                source_sents = [line.split("\t")[0].strip() for line in lines[i:i+args.batch_size]]
                
                try:
                    sorted_hypotheses, ref_scores = translator.translate_batch(source_sents, references)
                except RuntimeError:
                    print(f"{dataset=}")
                    print(f"{references=}")

                assert len(references) == len(source_sents) == len(sorted_hypotheses)
                
                for j in range(len(references)):
                    out.write(f"{format_json(args, source_sents[j], references[j], sorted_hypotheses[j], ref_score = ref_scores[j] if args.qe else ref_scores)}\n")
                    

                # debug
                if args.debug:
                    for i in range(len(references)):
                        debug.write(f"\n\n>>SRC: {source_sents[i]}\n")
                        debug.write(f">>REF: {references[i]}\n\n")
                        if args.qe:
                            debug.write(f">>REF SCORE: {ref_scores[i][1]}\n\n")
                        for j, h in enumerate(sorted_hypotheses[i]):
                            debug.write(f">>HYP_{j}: {h[1]}\t{h[0]}\n")
    
   

    
