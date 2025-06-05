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
import time


from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import random
import traceback

from mbrs.metrics import MetricBLEU, MetricChrF, MetricCOMET, MetricCOMETkiwi, MetricMetricX, MetricTER, MetricXCOMET
from mbrs.decoders import DecoderMBR, DecoderEnsembleMBR

MBR_ENSEMBLE = {
    "bleuf": ["bleu", "chrf"],
    "bleufpp": ["bleu", "chrf_pp"],
    "lex": ["bleu", "chrf", "ter"],
    "nolex_ref": ["comet", "metricx"],
    "comet_lex": ["comet", "bleu", "chrf"],
    "all_ref": ["bleu", "chrf", "ter", "comet", "metricx"],
    "qe": ["kiwi", "metricx"],
    
    "comet_bleu": ["comet", "bleu"],
}

MBR_MAP = {
    "bleu": MetricBLEU,
    "chrf": MetricChrF,
    "chrf_pp": MetricChrF,
    "ter": MetricTER, 
    "metricx": MetricMetricX,
    "comet": MetricCOMET,
    "kiwi": MetricCOMETkiwi,
    "xcomet": MetricXCOMET,
}

def init_mbr_metrics(metrics: list[str]) -> list["mbrs.metrics.Metric"]:
    return [init_mbr_metric(metric) for metric in metrics]

def init_mbr_metric(metric: str) -> "mbrs.metrics.Metric":
    match metric:
        case "bleu":
            # Removed multi-threading
            metric_cfg = MBR_MAP[metric].Config()
        case "chrf_pp":
            # Removed multi-threading
            metric_cfg = MBR_MAP[metric].Config(
                word_order=2,
            )
        case "chrf":
            metric_cfg = MBR_MAP[metric].Config(
                fastchrf=True,
            )
        case "ter":
            metric_cfg = MBR_MAP[metric].Config()
        case "comet":
            metric_cfg = MBR_MAP[metric].Config(
                model="Unbabel/wmt22-comet-da",
                batch_size=64,
                fp16=True,
            )
        case "kiwi":
            metric_cfg = MBR_MAP[metric].Config(
                model="Unbabel/wmt22-cometkiwi-da",
                batch_size=64,
                fp16=True,
            )
        case "metricx":
            metric_cfg = MBR_MAP[metric].Config(
                model="google/metricx-24-hybrid-large-v2p6-bfloat16",
                batch_size=64,
                bf16=True,
            )
        case "xcomet":
            raise NotImplementedError
        case _:
            raise NotImplementedError 

    metric = MBR_MAP[metric](metric_cfg)
    return metric
    

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--slang", type = str, choices = ["nb", "nn", "en"], help = "source language")
    parser.add_argument("--tlang", type = str, choices = ["nb", "nn", "en"], help = "target language")
    parser.add_argument("--method", type = str, help = "Translation method (model)")
    parser.add_argument("--batch_size", type = int, default = 1, help = "Batch size")
    parser.add_argument("--output_dir", type = str, default = None, help = "Name of output directory")
    parser.add_argument("--output_suffix", type = str, default = "", help = "Additional suffix to the output directory")
    parser.add_argument("--seed", type = int, default = 73, help = "Random seed")
    parser.add_argument("--beams", type = int, default = 1, help = "Number of beams for beam search. Currently only used by normistral, default = 1 (greed)")

    #Generate synthetic data args (NOT USED IN THIS FILE)
    parser.add_argument("--dataset", type = str, default = None, help = "Path to dataset to load, only used in generate_synthetic_data.py")
    parser.add_argument("--scorer", choices = ["top", "sim", "both"], default = "sim", help = "Which scorer to use. 'sim': LABSE similarity scoring, 'top': naive choose top, 'both': create two files, one top and one sim")

    #parser.add_argument("--output", type = str, default = "hypotheses")

    #NorMistral args
    parser.add_argument("--n_shots", type = int, default = None)
    parser.add_argument("--normistral", type = str, choices = ["7", "11"], help = "Whether to use normistral 7b or 11b")
    parser.add_argument("--prompt_style", type = str, choices = ["basic", "instruct", "instruct_repeat", "basic_en", "instruct_en", "instruct_repeat_en"], default = "basic", help = "Which prompt formatting style to use. Basic = zero-shot, instruct = zero-shot with initial instruction line, instruct_repeat = only used for Nshot, repeat instruction before each shot.")
    parser.add_argument("--lora_merge", type = str, default = None, help = "Path to lora checkpoint to merge with main model. Won't be used if not provided")
    parser.add_argument("--lora", type = str, default = None, help = "Path to lora checkpoint. Won't be used if not provided.")
    parser.add_argument("--max_new_tokens", type = int, default = 256, help = "Max new number of tokens to generate for normistral, default = 256")
    parser.add_argument("--bf16", action = "store_true", help = "If provided, load normistral in bfloat16 instead of float16")

    #Fairseq args
    parser.add_argument("--spm", type=str, default = "", help = "Path to sentencepiece model")
    parser.add_argument("--model_dir", type=str, default = "", help = "Path to model dir with dict")
    parser.add_argument("--model", type=str, default = "", help = "Path to fairseq checkpoint")

    #fairseq multi args
    parser.add_argument("--encoder_langtok", type = str, choices = ["tgt", "src"], help = "Whether to use source side or target side encoder langtoken")
    parser.add_argument("--decoder_langtok", action = "store_true", help = "Whether to add a decoder langtoken")
    parser.add_argument("--tagged_bt", action = "store_true", help = "Whether to add tagged backtranslation tokens for compatability.")

    #Decoding args
    parser.add_argument("--sample_epsilon", type = float, default = None, help = "If provided, epsilon sampling is used")
    parser.add_argument("--mbr", type = str, default = None, choices = MBR_MAP.keys(), help = "If provided, use the specifided utility for minimum bayes risk decoding")
    parser.add_argument("--mbr_ensemble", type = str, default = None, choices = MBR_ENSEMBLE.keys(), help = "Which MBR ensembling technique to use. Won't be used if not provided.")
    parser.add_argument("--n_hyp", type = int, default = 5, help = "Number of beams")

    parser.add_argument("--no_eval", action = "store_true", help = "Don't run evaluation, intended for interactive use")
    
    args = parser.parse_args()

    # Set do_mbr to True if either of the mbr arguments are provided
    args.do_mbr = (args.mbr is not None) or (args.mbr_ensemble is not None)
    
    return args

class AbstractTranslator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"SYSTEM: Running on {self.device}", flush=True)

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

        # Load and merge lora checkpoint if provided
        if args.lora_merge is not None:
            peft_model = PeftModel.from_pretrained(self.model, args.lora_merge)
            self.model = peft_model.merge_and_unload()
            print(f"Successfully loaded and merged LORA checkpoint {args.lora_merge}")
        
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
    def translate(self, sentence: str) -> str:
        text = self.format_prompt(sentence)
        
        input_ids = self.tokenizer(text, return_tensors='pt').input_ids.to(self.device)
        prediction = self.model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=args.beams,
            #normistral 11B tokenizer stops at numbers when \n is eos_token_id
            eos_token_id=None if args.normistral == '11' else self.tokenizer('\n').input_ids
        )

        #Return newly generated tokens
        return self.tokenizer.decode(prediction[0, input_ids.size(1):]).strip()

    
    @torch.no_grad()
    def translate_batch(self, sentences: list[str]) -> list[str]:
        max_new_tokens = min(int(max([len(t) for t in self.tokenizer(sentences).input_ids]) * 1.5), args.max_new_tokens)

        texts = [self.format_prompt(sent) for sent in sentences]
        
        model_inputs = self.tokenizer(texts, return_tensors='pt', return_token_type_ids = False, padding = True).to(self.device)
        input_length = model_inputs.input_ids.shape[1]
            
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=args.beams,
            eos_token_id=None if args.normistral == '11' else self.tokenizer('\n').input_ids
        )
        
        #return self.tokenizer.batch_decode(generated_ids[:, input_length:])
        # Default to splitting at new line
        return [sent.split("\n")[0].strip() for sent in self.tokenizer.batch_decode(generated_ids[:, input_length:])]
        

class NorMistralTranslatorNShot(NorMistralBase):
    def __init__(self, args):
        super().__init__(args)

        # Print example few-shot prompt
        example_data = ["I am here\tJeg er her", "He is there\tHan er der", "We are here\tVi er her"]
        example_prompt = self.format_prompt([x.split("\t")[0] for x in example_data], [x.split("\t")[1] for x in example_data], "Where are we?", "Engelsk", "Bokmål")
        print(f"Example prompt:\n{'-'*40}\n{example_prompt}\n{'-'*40}", flush = True)
    

    def format_prompt(
        self,
        source_shots: list[str],
        target_shots: list[str],
        sentence: str,
        source_language: str,
        target_language: str,
    ):
        # As in "The unreasonable effectiveness of few-shot learning for machine translation"
        # (https://arxiv.org/abs/2302.01398)
        prompt_template = (
            "{source_language}: {source_text}\n{target_language}:{target_text}"
        )

        if self.args.prompt_style.endswith("en"):
            if source_language == "Engelsk":
                source_language = "English"
            elif target_language == "Engelsk":
                target_language = "English"
        
        examples = [
            prompt_template.format(
                source_language=source_language,
                target_language=target_language,
                source_text=source_text,
                target_text=" " + target_text
            )
            for i, (source_text, target_text) in enumerate(zip(source_shots, target_shots))
        ]
        few_shot_string = "\n\n".join(
            examples
        )  # Join the examples with two newlines (https://arxiv.org/abs/2302.01398)

        if self.args.prompt_style == "instruct":
            # Instruction once before shots and once before the final prompt
            instruction = f"Oversett dette fra {source_language} til {target_language}:"
            
            final_prompt_part = f"{instruction}\n{source_language}: {sentence}\n{target_language}:"
            prompt = f"{instruction}\n{few_shot_string}\n\n{final_prompt_part}"

        
        elif self.args.prompt_style == "instruct_en":
            # Instruction once before shots and once before the final prompt
            instruction = f"Translate this from {source_language} to {target_language}:"
            
            final_prompt_part = f"{instruction}\n{source_language}: {sentence}\n{target_language}:"
            prompt = f"{instruction}\n{few_shot_string}\n\n{final_prompt_part}"

        
        elif self.args.prompt_style == "instruct_repeat":
            # Instruction before each shot and before the final prompt
            instruction = f"Oversett dette fra {source_language} til {target_language}:"
            examples = [f"{instruction}\n{ex}" for ex in examples]
            few_shot_string = "\n\n".join(examples)
            
            final_prompt_part = f"{instruction}\n{source_language}: {sentence}\n{target_language}:"
            prompt = f"{few_shot_string}\n\n{final_prompt_part}"

        
        elif self.args.prompt_style == "instruct_repeat_en":
            # Instruction before each shot and before the final prompt
            instruction = f"Translate this from {source_language} to {target_language}:"
            examples = [f"{instruction}\n{ex}" for ex in examples]
            few_shot_string = "\n\n".join(examples)
            
            final_prompt_part = f"{instruction}\n{source_language}: {sentence}\n{target_language}:"
            prompt = f"{few_shot_string}\n\n{final_prompt_part}"
            
        else:
            # No instruction
            final_prompt_part = f"{source_language}: {sentence}\n{target_language}:"
            prompt = f"{few_shot_string}\n\n{final_prompt_part}"
    
        return prompt
    
    def sample_random_examples(self, dataset: list[dict], example_index: int, n_shots: int) -> list[str]:
        # Sample n_shots different examples from the dataset (excluding the example at example_index)
        sequence = list(range(0, example_index)) + list(
            range(example_index + 1, len(dataset))
        )
        random_indices = random.sample(sequence, n_shots)
        return [dataset[j] for j in random_indices]
    
    @torch.no_grad()
    def translate(self, sentence: str, dataset, index) -> str:

        shots = self.sample_random_examples(dataset, index, self.args.n_shots)
        
        source_texts = [s.split("\t")[0] for s in shots]
        target_texts = [s.split("\t")[1] for s in shots]
        
        #source_texts.append(sentence)
        #target_texts.append("")

        text = self.format_prompt(source_texts, target_texts, sentence, self.LANGUAGE_IDS[self.args.slang], self.LANGUAGE_IDS[self.args.tlang])

        input_ids = self.tokenizer(text, return_tensors='pt').input_ids.to(self.device)
        prediction = self.model.generate(
            input_ids,
            max_new_tokens=64,
            do_sample=False,
            num_beams=args.beams,
            eos_token_id=None if args.normistral == '11' else self.tokenizer('\n').input_ids,
        )
        return self.tokenizer.decode(prediction[0, input_ids.size(1):]).strip()

    
    @torch.no_grad()
    def translate_batch(self, sentences: list[str], dataset, indices) -> list[str]:
        max_new_tokens = min(int(max([len(t) for t in self.tokenizer(sentences).input_ids]) * 1.5), args.max_new_tokens)

        texts = []
        
        for i in range(len(sentences)):
            shots = self.sample_random_examples(dataset, indices[0], self.args.n_shots)
            source_texts = [s.split("\t")[0] for s in shots]
            target_texts = [s.split("\t")[1] for s in shots]

            text = self.format_prompt(source_texts, target_texts, sentences[i], self.LANGUAGE_IDS[self.args.slang], self.LANGUAGE_IDS[self.args.tlang])

            texts.append(text)
        
        model_inputs = self.tokenizer(texts, return_tensors='pt', return_token_type_ids = False, padding = True).to(self.device)
        input_length = model_inputs.input_ids.shape[1]
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=args.beams,
            eos_token_id=None if args.normistral == '11' else self.tokenizer('\n').input_ids,
        )
        return [sent.split("\n")[0].strip() for sent in self.tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens = True)]


class AlmaTranslator(AbstractTranslator):
    def __init__(self, args):
        super().__init__(args)

        #self.LANGUAGE_IDS = {
        #    "en": "Engelsk",
        #    "nb": "Bokmål",
        #    "nn": "Nynorsk"
        #}

        self.LANGUAGE_IDS = {
            "en": "English",
            "nb": "Norwegian",
            "nn": "Norwegian"
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            f"haoranxu/X-ALMA-13B-Group1",
            padding_side='left'
        )

        #Check tokenizer pad side, sometimes it does not change the pad side during init
        print(f"Tokenizer padding side: {self.tokenizer.padding_side}")
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"
            print(f"Set tokenizer padding side to \'left\'")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Added pad token: {self.tokenizer.pad_token}, id = {self.tokenizer.pad_token_id}")
            

        self.model = AutoModelForCausalLM.from_pretrained(
            f"haoranxu/X-ALMA-13B-Group1",
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        ).to(self.device).eval()

    
    def format_prompt(self, sentence: str) -> str:
        prompt = f"""Translate this from {self.LANGUAGE_IDS[self.args.slang]} to {self.LANGUAGE_IDS[self.args.tlang]}: 
{self.LANGUAGE_IDS[self.args.slang]}: {sentence}
{self.LANGUAGE_IDS[self.args.tlang]}:"""

        #apply chat template
        chat_style_prompt = [{"role": "user", "content": prompt}]
        prompt = self.tokenizer.apply_chat_template(
            chat_style_prompt,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    @torch.no_grad()
    def translate(self, sentence: str) -> str:
        prompt = self.format_prompt(sentence)

        # from huggingface
        #input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=40, truncation=True).input_ids.to(self.device)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        #from huggingface
        #generated_ids = model.generate(input_ids=input_ids, num_beams=5, max_new_tokens=20, do_sample=True, temperature=0.6, top_p=0.9)
        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens = args.max_new_tokens,
            do_sample = False,
            num_beams = args.beams,
        )

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    @torch.no_grad()
    def translate_batch(self, sentences: list[str]) -> list[str]:
        max_new_tokens = min(int(max([len(t) for t in self.tokenizer(sentences).input_ids]) * 1.5), args.max_new_tokens)

        texts = [self.format_prompt(sent) for sent in sentences]
        
        model_inputs = self.tokenizer(texts, return_tensors='pt', return_token_type_ids = False, padding = True).to(self.device)
        #input_length = model_inputs.input_ids.shape[1]
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=args.beams,
        )
        
        #return self.tokenizer.batch_decode(generated_ids[:, input_length:])
        # Default to splitting at new line
        return [sent.split("\n")[0].strip() for sent in self.tokenizer.batch_decode(generated_ids, skip_special_tokens = True)]
        

class NorT5Translator(AbstractTranslator):
    def __init__(self, args):
        super().__init__(args)
        
        self.tokenizer = AutoTokenizer.from_pretrained("ltg/nort5-base-en-no-translation")
        self.cls_index = self.tokenizer.convert_tokens_to_ids("[CLS]")
        self.sep_index = self.tokenizer.convert_tokens_to_ids("[SEP]")
        self.eos_index = self.tokenizer.convert_tokens_to_ids("[EOS]")
        self.pad_index = self.tokenizer.convert_tokens_to_ids("[PAD]")
        self.eng_index = self.tokenizer.convert_tokens_to_ids(">>eng<<")
        self.nob_index = self.tokenizer.convert_tokens_to_ids(">>nob<<")
        self.nno_index = self.tokenizer.convert_tokens_to_ids(">>nno<<")

        self.LANGUAGE_IDS = {
            "en": self.eng_index,
            "nb": self.nob_index,
            "nn": self.nno_index
        }

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "ltg/nort5-base-en-no-translation", 
            trust_remote_code=True, 
        ).to(self.device).eval()

        print(f"Sucessfully loaded the model to the memory")

    class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
        def __init__(self, penalty: float, model):
            last_bias = model.classifier.nonlinearity[-1].bias.data
            last_bias = torch.nn.functional.log_softmax(last_bias, dim = 0)
            self.penalty = penalty * (last_bias - last_bias.max())
    
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            penalized_score = torch.gather(scores + self.penalty.unsqueeze(0).to(input_ids.device), 1, input_ids).to(scores.dtype)
            scores.scatter_(1, input_ids, penalized_score)
            return scores
    
    def translate(self, sentence: str) -> str:
        source = [s.strip() for s in sentence.split('\n')]
        source_subwords = self.tokenizer(source).input_ids
        source_subwords = [[self.cls_index, self.LANGUAGE_IDS[args.tlang], self.LANGUAGE_IDS[args.slang]] + s + [self.sep_index] for s in source_subwords]
        source_subwords = [torch.tensor(s) for s in source_subwords]
        source_subwords = torch.nn.utils.rnn.pad_sequence(source_subwords, batch_first=True, padding_value=self.pad_index)
        source_subwords = source_subwords[:, :512].to(self.device)

        def generate(model, **kwargs):
            with torch.inference_mode():
                with torch.autocast(enabled=self.device != "cpu", device_type="cuda", dtype=torch.bfloat16):
                    return model.generate(**kwargs)

        generate_kwargs = dict(
            input_ids=source_subwords,
            attention_mask=(source_subwords != self.pad_index).long(),
            max_new_tokens = 512-1,
            num_beams=args.beams,
            length_penalty=1.6,
            early_stopping=True,
            do_sample=False,
            use_cache=True,
            logits_processor=[self.RepetitionPenaltyLogitsProcessor(0.5, self.model), transformers.LogitNormalization()]
        )
        output = generate(self.model, **generate_kwargs).tolist()
        paragraphs = [self.tokenizer.decode(c, skip_special_tokens=True).strip() for c in output]
        translation = '\n'.join(paragraphs)

        return translation


class FairseqMonoTranslator(AbstractTranslator):
    def __init__(self, args):
        super().__init__(args)

        model_dir = Path(args.model_dir)
        model_path = model_dir / args.model
        spm_model = model_dir / args.spm

        print(f"Decoding strategy: {'MBR' if args.do_mbr else 'BEAM SEARCH'}")
        
        # Initialize the Fairseq parser and add arguments as done in fairseq-interactive
        parser = options.get_interactive_generation_parser()
        if args.do_mbr:
            self.fairseq_args = options.parse_args_and_arch(parser, input_args=[
                str(model_dir),
                '--path', str(model_path),
                '--task', 'epsilon_translation',
                '--source-lang', args.slang,
                '--target-lang', args.tlang,
                '--bpe', 'sentencepiece',
                '--sentencepiece-model', str(spm_model),
                '--sacrebleu',
                '--remove-bpe', 'sentencepiece',
                '--user-dir', '../custom_fairseq',
                '--seed', str(args.seed),
                '--beam', str(args.n_hyp),
                '--sampling',
                '--nbest', str(args.n_hyp),
                '--sampling-epsilon', str(0.02),
                '--fp16',
            ])

        else:
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

        if args.do_mbr:
            if args.mbr_ensemble is not None:
                metrics = init_mbr_metrics(MBR_ENSEMBLE[args.mbr_ensemble])
                decoder_cfg = DecoderEnsembleMBR.Config()
                self.decoder = DecoderEnsembleMBR(decoder_cfg, metrics)
            else:
                metric = init_mbr_metric(args.mbr)
                decoder_cfg = DecoderMBR.Config()
                self.decoder = DecoderMBR(decoder_cfg, metric)

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

        if self.args.do_mbr:
            translation_tokens = [t['tokens'] for t in translations[0]]
            
            # Convert tokens to text
            decoded_output = [self.task.tgt_dict.string(t) for t in translation_tokens]
            # Detokenize output using SentencePiece
            hypotheses = [self.bpe.decode(d) for d in decoded_output]
            
            mbr_decoded_output = self.decoder.decode(hypotheses, hypotheses, source=sentence, nbest=1)
            
            return mbr_decoded_output.sentence

        #Standard beam search
        else:
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
            if self.args.do_mbr:
                output_tokens = [t['tokens'] for t in translation]

                #decoded_output = [self.task.tgt_dict.string(t) for t in output_tokens]
                #hypotheses = [self.bpe.decode(d) for d in decoded_output]
                hypotheses = [self.bpe.decode(self.task.tgt_dict.string(t)) for t in output_tokens]
                mbr_decoded_output = self.decoder.decode(hypotheses, hypotheses, source=sentences[i], nbest=1)
                
                results.append(mbr_decoded_output.sentence[0])
            else:
                output_tokens = translation[0]['tokens']
                
                #decoded_output = self.task.tgt_dict.string(output_tokens)
                #decoded_output = self.bpe.decode(decoded_output)
                decoded_output = self.bpe.decode(self.task.tgt_dict.string(output_tokens))
        
                results.append(decoded_output)
    
        return results


class FairseqMultiTranslator(AbstractTranslator):
    def __init__(self, args):
        super().__init__(args)
        
        model_dir = Path(args.model_dir)
        model_path = model_dir / args.model
        spm_model = model_dir / args.spm

        #'--lang-pairs', "nb-en,en-nb,nn-en,en-nn,nb-nn,nn-nb"
        
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
        self.fairseq_args = options.parse_args_and_arch(parser, input_args=input_args)
        

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

    @torch.no_grad()
    def translate(self, sentence: str) -> str:
        # Tokenize input text using SentencePiece
        tokenized_input = self.bpe.encode(sentence)
    
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
                line=self.bpe.encode(sentence),
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
    """Get all relevant test sets from the specified source path"""
    source_path = Path("path/to/test_sets")
    datasets = []
    #Swap direction if 'slang' is the target language of the file, and vice versa
    swap = False
    
    for subfolder in source_path.iterdir():
        for file in subfolder.iterdir():
            #Find source and target language from file extension
            s, t = file.name.split(".")[-1].split("-")
            #Correct way
            if s == slang and t == tlang:
                datasets.append(file)
                swap = False
            #Wrong way
            elif s == tlang and t == slang:
                datasets.append(file)
                swap = True

    #Return list of datasets with a boolean that tells if the dataset is the right direction
    return datasets, swap

def get_gpu_vram_info():
    name = "CPU"
    VRAM = "0"
    if torch.cuda.is_available():
        # Get number of GPUs
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            # Get the name of the GPU
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            name = torch.cuda.get_device_name(i)
            # Get the total VRAM for the GPU
            vram_in_bytes = torch.cuda.get_device_properties(i).total_memory
            vram_in_gb = vram_in_bytes / (1024 ** 3)
            VRAM = vram_in_gb
            print(f"Total VRAM: {vram_in_gb:.2f} GB")
    else:
        print("No GPU detected.")

    return name, VRAM

def set_seed(seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    gpu_name, gpu_vram = get_gpu_vram_info()
    set_seed(73)
    
    print(f"\n{' '.join(sys.argv)}\n")
    args = parse_args()

    if args.method == "nort5":
        translator = NorT5Translator(args)
    elif args.method == "normistral":
        translator = NorMistralTranslator(args)
    elif args.method == "normistralNshot":
        translator = NorMistralTranslatorNShot(args)
    elif args.method == "fairseq_mono":
        translator = FairseqMonoTranslator(args)
    elif args.method == "fairseq_multi":
        translator = FairseqMultiTranslator(args)
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    if not args.no_eval:
        if args.output_dir is None:
            b_string = '' if 'normistral' not in args.method else f"_{args.normistral}b"
            lora_string = f"_LORA_{args.lora.split('/')[-2].split('_')[-1]}" if args.lora is not None else ''
            #big_string = '_big' if 'big' in args.model_dir else ''
            
            f_offset = 4 if args.method == "fairseq_multi" else 3
            fairseq_string = f"_{'_'.join(args.model_dir.split('/')[-1].split('_')[f_offset:])}" if args.model_dir is not None else ""
            if fairseq_string == "_":
                fairseq_string = ""
            
            output_dir = Path("NEW_TEST_SET") / f"{args.method}{b_string}{lora_string}{fairseq_string}__{args.slang}_{args.tlang}{args.output_suffix}"
        else:
            output_dir = Path("NEW_TEST_SET") / f"{args.output_dir}{args.output_suffix}"
        output_dir.mkdir()
        
        datasets, swap = get_datasets(args.slang, args.tlang)
        for dataset in datasets:
            with open(dataset, "r", encoding = "utf-8") as test_file:
                lines = test_file.read().strip().split("\n")
    
            output_file = output_dir / f"{dataset.name}.hyp"
            ref_file = output_dir / f"{dataset.name}.ref"
    
            # Time it
            start_time = time.time()
            
            # normistral with N-shot prompting
            if args.method == "normistralNshot" and args.n_shots is not None and args.batch_size == 1:
                with open(output_file, "w", encoding = "utf-8") as hyp, \
                    open(ref_file, "w", encoding = "utf-8") as ref:
                        
                    for i, line in enumerate(tqdm(lines), mininterval = 5):
                        reference = line.split("\t")[0 if swap else 1].strip()
                        
                        source_sent = line.split("\t")[1 if swap else 0].strip()
                        hypothesis = translator.translate(source_sent, lines, i)
                        
                        ref.write(f"{reference}\n")
                        hyp.write(f"{hypothesis}\n")
    
            # All models with batch size 1
            elif args.batch_size == 1:
                with open(output_file, "w", encoding = "utf-8") as hyp, \
                    open(ref_file, "w", encoding = "utf-8") as ref:
                    for line in tqdm(lines, mininterval = 5):
                        reference = line.split("\t")[0 if swap else 1].strip()
    
                        source_sent = line.split("\t")[1 if swap else 0].strip()
                        hypothesis = translator.translate(source_sent)
                        
                        ref.write(f"{reference}\n")
                        hyp.write(f"{hypothesis}\n")
    
    
            #Batch size above 1 for nort5
            elif args.batch_size > 1 and args.method == "nort5":
                with open(output_file, "w", encoding = "utf-8") as hyp, \
                    open(ref_file, "w", encoding = "utf-8") as ref:
                    for i in tqdm(range(0, len(lines), args.batch_size), mininterval = 5):
                        references = "\n".join([line.split("\t")[0 if swap else 1].strip() for line in lines[i:i+args.batch_size]])
    
                        source_sents = "\n".join([line.split("\t")[1 if swap else 0].strip() for line in lines[i:i+args.batch_size]])
                        hypotheses = translator.translate(source_sents)
                        
                        for r in references.split("\n"):
                            ref.write(f"{r}\n")
                        for h in hypotheses.split("\n"):
                            hyp.write(f"{h}\n")
    
            #Batch size above 1 and normistral N-shot
            elif args.batch_size > 1 and args.method == "normistralNshot" and args.n_shots is not None:
                print(f"Executing Normistal N-shot with high batch size")
                with open(output_file, "w", encoding = "utf-8") as hyp, \
                    open(ref_file, "w", encoding = "utf-8") as ref:
                        
                    for i in tqdm(range(0, len(lines), args.batch_size), mininterval = 5):
                        references = [line.split("\t")[0 if swap else 1].strip() for line in lines[i:i+args.batch_size]]
                        
                        source_sents = [line.split("\t")[1 if swap else 0].strip() for line in lines[i:i+args.batch_size]]
                        try:
                            hypotheses = translator.translate_batch(source_sents, lines, list(range(i, i+args.batch_size)))
                        except RuntimeError as e:
                            print(f"An error occurred: {e}, at batch {i}-{i+args.batch_size}")
                            traceback.print_exc()
                            #print(f"{dataset=}")
                            #print(f"{references=}")
                        for r in references:
                            ref.write(f"{r}\n")
                        for h in hypotheses:
                            hyp.write(f"{h}\n")
    
            #All methods with batch size above 1 except except normistralNshot and nort5
            else:
                with open(output_file, "w", encoding = "utf-8") as hyp, \
                    open(ref_file, "w", encoding = "utf-8") as ref:
                    for i in tqdm(range(0, len(lines), args.batch_size), mininterval = 5):
                        references = [line.split("\t")[0 if swap else 1].strip() for line in lines[i:i+args.batch_size]]
                        
                        source_sents = [line.split("\t")[1 if swap else 0].strip() for line in lines[i:i+args.batch_size]]
                        #try:
                        hypotheses = translator.translate_batch(source_sents)
                        #except RuntimeError:
                        #    print(f"{dataset=}")
                        #    print(f"{references=}")
                        for r in references:
                            ref.write(f"{r}\n")
                        for h in hypotheses:
                            hyp.write(f"{h}\n")
    
    
    
            # Append runtime to hyp file
            with open(output_file, "a", encoding = "utf-8") as hyp:
                # Calculate elapsed time and print average time per sentence
                elapsed_time = time.time() - start_time
                average_time_per_sentence = elapsed_time / len(lines)
                
                print(f"{dataset.name}: Average time per sentence: {average_time_per_sentence:.4f} seconds.")
                # Write to file, should be ingored when computing metrics
                hyp.write(f"##{average_time_per_sentence}\t{args.batch_size}\t{gpu_name}\t{gpu_vram}\n")
