import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

from mbrs.metrics import MetricBLEU, MetricChrF, MetricCOMET
from mbrs.decoders import DecoderMBR

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
    #parser.add_argument("--dataset", type = str, default = "test_3500.en_nb")
    #parser.add_argument("--output", type = str, default = "hypotheses")

    #NorMistral args
    parser.add_argument("--n_shots", type = int, default = None)
    parser.add_argument("--normistral", type = str, choices = ["7", "11"], help = "Whether to use normistral 7b or 11b")
    parser.add_argument("--prompt_style", type = str, choices = ["basic", "instruct"], default = "basic", help = "Which prompt formatting style to use. Basic = zero-shot, instruct = zero-shot with initial instruction line")
    parser.add_argument("--lora", type = str, default = None, help = "Path to lora checkpoint. Won't be used if not provided.")
    parser.add_argument("--lora_merge", type = str, default = None, help = "Path to lora checkpoint to merge with main model. Won't be used if not provided")
    parser.add_argument("--max_new_tokens", type = int, default = 256, help = "Max new number of tokens to generate for normistral, default = 256")
    parser.add_argument("--bf16", action = "store_true", help = "If provided, load normistral in bfloat16 instead of float16")

    #Fairseq args
    parser.add_argument("--spm", type=str, default = "", help = "Path to sentencepiece model")
    parser.add_argument("--model_dir", type=str, default = "", help = "Path to model dir with dict")
    parser.add_argument("--model", type=str, default = "", help = "Path to fairseq checkpoint")

    #fairseq multi args
    parser.add_argument("--encoder_langtok", type = str, choices = ["tgt", "src"], help = "Whether to use source side or target side encoder langtoken")

    #Decoding args
    parser.add_argument("--sample_epsilon", type = float, default = None, help = "If provided, epsilon sampling is used")
    parser.add_argument("--mbr", action = "store_true", help = "If provided, use minimum bayes risk decoding")
    parser.add_argument("--n_hyp", type = int, default = 5, help = "Number of beams")

    return parser.parse_args()

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
            
            print("Model is a peft model: ", end ="")
            print(isinstance(self.model, PeftModel))
            
            print(f"Successfully loaded and merged LORA checkpoint {args.lora_merge}")
        
        # Load lora checkpoint if provided
        if args.lora is not None:
            self.model = PeftModel.from_pretrained(self.model, args.lora)
            
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

    def format_prompt(
        self,
        source_texts: list[str],
        target_texts: list[list[str]],
        source_language: str,
        target_language: str,
    ):
        # As in "The unreasonable effectiveness of few-shot learning for machine translation"
        # (https://arxiv.org/abs/2302.01398)
        prompt_template = (
            "{source_language}: {source_text}\n{target_language}:{target_text}"
        )
    
        examples = [
            prompt_template.format(
                source_language=source_language,
                target_language=target_language,
                source_text=source_text,
                target_text=" " + target_text
                if i < len(source_texts) - 1
                else "",  # Add space before target text (except for the last example,
                # which should be empty)
            )
            for i, (source_text, target_text) in enumerate(zip(source_texts, target_texts))
        ]
        prompt = "\n\n".join(
            examples
        )  # Join the examples with two newlines (https://arxiv.org/abs/2302.01398)
    
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
        
        source_texts.append(sentence)
        target_texts.append("")

        text = self.format_prompt(source_texts, target_texts, self.LANGUAGE_IDS[self.args.slang], self.LANGUAGE_IDS[self.args.tlang])

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

            source_texts.append(sentences[i])
            target_texts.append("")

            text = self.format_prompt(source_texts, target_texts, self.LANGUAGE_IDS[self.args.slang], self.LANGUAGE_IDS[self.args.tlang])

            texts.append(text)
        
        #input_ids = self.tokenizer(texts, return_tensors='pt').input_ids.to(self.device)
        model_inputs = self.tokenizer(texts, return_tensors='pt', return_token_type_ids = False, padding = True).to(self.device)
        input_length = model_inputs.input_ids.shape[1]
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=args.beams,
            eos_token_id=None if args.normistral == '11' else self.tokenizer('\n').input_ids,
        )
        #return self.tokenizer.batch_decode(generated_ids[:, input_length:])
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
            #torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
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
            num_beams=5,
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

        print(f"Decoding strategy: {'MBR' if args.mbr else 'BEAM SEARCH'}")
        
        # Initialize the Fairseq parser and add arguments as done in fairseq-interactive
        parser = options.get_interactive_generation_parser()
        if args.mbr:
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
                '--beam', '5',
                '--user-dir', '../custom_fairseq'
            ])
        
        
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
    def translate(self, sentence: str) -> str:
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

        if args.mbr:
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
        translations = self.task.inference_step(self.generator, self.models, sample)
    
        # Decode the output for each sample in the batch
        results = []
        for i, translation in enumerate(translations):
            if args.mbr:
                output_tokens = [t['tokens'] for t in translation]

                decoded_output = [self.task.tgt_dict.string(t) for t in output_tokens]
                hypotheses = [self.bpe.decode(d) for d in decoded_output]
                mbr_decoded_output = self.decoder.decode(hypotheses, hypotheses, source=sentences[i], nbest=1)
                
                results.append(mbr_decoded_output.sentence[0])
            else:
                output_tokens = translation[0]['tokens']
                decoded_output = self.task.tgt_dict.string(output_tokens)
        
                decoded_output = self.bpe.decode(decoded_output)
        
                results.append(decoded_output)
    
        return results


class FairseqMultiTranslator(AbstractTranslator):
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
            '--task', 'translation_multi_simple_epoch',
            '--source-lang', args.slang,
            '--target-lang', args.tlang,
            '--bpe', 'sentencepiece',
            '--sentencepiece-model', str(spm_model),
            '--sacrebleu',
            '--remove-bpe', 'sentencepiece',
            '--beam', '5',
            '--encoder-langtok', args.encoder_langtok,
            '--decoder-langtok',
            '--lang-pairs', "nb-en,en-nb,nn-en,en-nn,nb-nn,nn-nb"
        ])
        

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



if __name__ == "__main__":
    random.seed(73)
    print(sys.argv)
    args = parse_args()

    if args.method == "nort5":
        translator = NorT5Translator(args)
    elif args.method == "normistral":
        translator = NorMistralTranslator(args)
    elif args.method == "normistralNshot":
        translator = NorMistralTranslatorNShot(args)
    elif args.method == "alma":
        translator = AlmaTranslator(args)
    elif args.method == "fairseq_mono":
        translator = FairseqMonoTranslator(args)
    elif args.method == "fairseq_multi":
        translator = FairseqMultiTranslator(args)
    else:
        raise ValueError(f"Unsupported method: {args.method}")


    dataset = Path("path/to/dataset")
    with open(dataset, "r", encoding = "utf-8") as test_file:
        lines = test_file.read().strip().split("\n")

    output_file = Path(f"SET/OUTPUT_PATH/{args.output_suffix}.hyp")

    with open(output_file, "w", encoding = "utf-8") as hyp:
        for i in tqdm(range(0, len(lines), args.batch_size)):
            references = [line.split("\t")[1].strip() for line in lines[i:i+args.batch_size]]
            
            source_sents = [line.split("\t")[0].strip() for line in lines[i:i+args.batch_size]]
            try:
                hypotheses = translator.translate_batch(source_sents)
            except RuntimeError:
                print(f"{dataset=}")
                print(f"{references=}")
            for h in hypotheses:
                hyp.write(f"{h}\n")
