""" Evaluate external models (models I did not make) """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Set TOKENIZERS_PARALLELISM so it doesn't annoy us
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable CUDA Launch Blocking to help debug CUDA errors
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import random
import sys
import requests
import html
import time
import asyncio

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from transformers.generation import LogitsProcessor
#from peft import PeftModel

import sentencepiece as spm

from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

"""from mbrs.metrics import MetricBLEU, MetricChrF, MetricCOMET, MetricCOMETkiwi, MetricMetricX, MetricTER, MetricXCOMET
from mbrs.decoders import DecoderMBR, DecoderEnsembleMBR"""


NLLB_MODEL_IDS = {
    "3.3B" : "facebook/nllb-200-3.3B",
    "1.3B" : "facebook/nllb-200-1.3B",
    "1.3B_distil" : "facebook/nllb-200-distilled-1.3B",
    "600M_distil" : "facebook/nllb-200-distilled-600M",
}
    
OPUS_MODEL_IDS = {
    "opus_en_gmq" : "Helsinki-NLP/opus-mt-tc-big-en-gmq",
    "opus_gmq_en" : "Helsinki-NLP/opus-mt-tc-big-gmq-en",
    "opus_multi_gmq" : "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmq",
    "opus_gmq_multi" : "Helsinki-NLP/opus-mt-tc-bible-big-gmq-deu_eng_fra_por_spa",
    "opus_en_nor" : "jkorsvik/opus-mt-eng-nor",
    "hplt_en_nn": "HPLT/translate-en-nn-v1.0-hplt_opus",
    "opus_nor_nor": "Helsinki-NLP/opus-mt-no-no",
}

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--slang", type = str, choices = ["nb", "nn", "en"], help = "source language")
    parser.add_argument("--tlang", type = str, choices = ["nb", "nn", "en"], help = "target language")
    parser.add_argument("--method", type = str, help = "Translation method (model)")
    parser.add_argument("--batch_size", type = int, default = 1, help = "Batch size")
    parser.add_argument("--output_dir", type = str, default = None, help = "Name of output directory")
    parser.add_argument("--output_suffix", type = str, default = "", help = "Additional suffix to the output directory")
    parser.add_argument("--seed", type = int, default = 73, help = "Random seed")
    parser.add_argument("--beams", type = int, default = 5, help = "Number of beams for beam search. default = 5")
    #parser.add_argument("--dataset", type = str, default = "test_3500.en_nb")
    #parser.add_argument("--output", type = str, default = "hypotheses")

    # Google args
    parser.add_argument("--google_method", type = str, choices = ["api", "deep", "py"], help = "Which google translation method to use")
    
    # NLLB HF args
    parser.add_argument("--nllb_model", type = str, default = None, choices = NLLB_MODEL_IDS.keys(), help = "WHich NLLB model to use")

    # Opus HF args
    parser.add_argument("--opus_model", type = str, default = None, choices = OPUS_MODEL_IDS.keys(), help = "WHich OPUS HF model to use")

    # Alma args
    parser.add_argument("--alma_sample", action = "store_true", help = "Whether or not to use default alma generation config")
    
    #Fairseq args
    parser.add_argument("--spm", type=str, default = "", help = "Path to sentencepiece model")
    parser.add_argument("--model_dir", type=str, default = "", help = "Path to model dir with dict")
    parser.add_argument("--model", type=str, default = "", help = "Path to fairseq checkpoint")
    parser.add_argument("--dict", type=str, default = None, help = "Path to fixed fairseq dictionary")
    parser.add_argument("--fp16", action = "store_true", help = "Load model in fp16")
    parser.add_argument("--lenpen", type = float, default = None, help = "Length penalty")
    parser.add_argument("--no_repeat_ngram_size", type = int, default = None, help = "No repeating ngrams")

    #fairseq multi args
    parser.add_argument("--encoder_langtok", type = str, choices = ["tgt", "src"], help = "Whether to use source side or target side encoder langtoken")

    parser.add_argument("--bf16", action = "store_true", help = "Use bf16, unsure if supported")
    parser.add_argument("--max_new_tokens", type = int, default = 256, help = "Max new number of tokens to generate for normistral, default = 256")
    parser.add_argument("--start_dataset", type = int, default = 0, help = "Index position for which dataset to start at")

    parser.add_argument("--no_eval", action = "store_true", help = "Do not run evaluation on test sets, intended for using the translator interactively")
    
    args = parser.parse_args()

    
    return args




class AbstractTranslator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"SYSTEM: Running on {self.device}", flush=True)

    def translate(self, sentence: str) -> str:
        raise NotImplementedError



class GoogleTranslatorDeep(AbstractTranslator):
    def __init__(self, args):
        super().__init__(args)
        
        from deep_translator import GoogleTranslator
        from deep_translator.exceptions import TranslationNotFound
        self.translation_exception = TranslationNotFound
        self.google_translator = GoogleTranslator

        self.LANGUAGE_IDS = {
            "en": "en",
            "nb": "no",
            "nn": "no",
        }

    def translate(self, sentence: str) -> str:
        translated = self.google_translator(source=self.LANGUAGE_IDS[self.args.slang], target=self.LANGUAGE_IDS[self.args.tlang]).translate(sentence)

        return translated

    def translate_batch(self, sentences: list[str]) -> list[str]:
        try:
            translated = self.google_translator(source=self.LANGUAGE_IDS[self.args.slang], target=self.LANGUAGE_IDS[self.args.tlang]).translate_batch(sentences)
        except self.translation_exception as e:
            print(f"No translation found for {sentences}\nresult: {translated}")
        
        return translated

class GoogleTranslatorPy(AbstractTranslator):
    def __init__(self, args):
        super().__init__(args)
        
        from googletrans import Translator
        self.google_translator = Translator

        self.LANGUAGE_IDS = {
            "en": "en",
            "nb": "no",
            "nn": "no",
        }

    async def translate(self, sentence: str) -> str:
        async with self.google_translator() as translator:
            result = await translator.translate(sentence, src=self.LANGUAGE_IDS[self.args.slang], dest=self.LANGUAGE_IDS[self.args.tlang])
            
            return result.text

    async def translate_batch(self, sentences: list[str]) -> list[str]:
        async with self.google_translator() as translator:
            translations = await translator.translate(sentences, src=self.LANGUAGE_IDS[self.args.slang], dest=self.LANGUAGE_IDS[self.args.tlang])

            return [t.text for t in translations]

class GoogleTranslatorAPI(AbstractTranslator):
    def __init__(self, args):
        super().__init__(args)

        self.LANGUAGE_IDS = {
            "en": "en",
            "nb": "no",
            "nn": "no",
        }
        
        # https://github.com/ssut/py-googletrans/issues/268#issuecomment-2647612796
        self.api_key = "API_KEY"

    def translate(self, sentence: str) -> str:
        # Define the API endpoint and API key
        url = "https://translate-pa.googleapis.com/v1/translateHtml"
        
        # Define the headers and payload
        headers = {
            "Content-Type": "application/json+protobuf",
            "X-Goog-API-Key": self.api_key,
        }
        
        payload = [
            [[sentence], self.LANGUAGE_IDS[self.args.slang], self.LANGUAGE_IDS[self.args.tlang]],
            "wt_lib"
        ]
        
        # Make the POST request
        response = requests.post(url, headers=headers, json=payload)
    
        
        # Print the response
        if response.status_code == 200:
            resp = response.json()
            text = resp[0][0]
            
            return html.unescape(text)
            #print("Translation:", response.json())
        else:
            return None
            #print("Error:", response.status_code, response.text)

class OpusTranslator_HF(AbstractTranslator):
    # Requires transformers 1.36 < x < 1.40
    def __init__(self, args):
        super().__init__(args)

        self.LANGUAGE_IDS = {
            "en": ">>eng<<",
            "nb": ">>nob<<",
            "nn": ">>nno<<",
        }

        model_path = OPUS_MODEL_IDS[self.args.opus_model]
        print(f"{model_path=}", flush = True)

        # Determine target token
        target_group = self.args.opus_model.split("_")[-1].strip()
        # Monolingual
        if target_group in ["en", "nn", "nb"]:
            self.target_label = ""

        # Translate into NNO or NOB
        elif target_group in ["gmq", "nor"]:
            self.target_label = self.LANGUAGE_IDS[self.args.tlang]
            self.target_label += " "

        # Translate into English
        elif target_group == "multi":
            self.target_label = self.LANGUAGE_IDS["en"]
            self.target_label += " "

        else:
            raise ValueError(f"Unsupported target_group {target_group}")

        print(f"Prepending {self.target_label} to source sentences")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype = torch.float16 if args.fp16 else "auto",
        ).to(self.device).eval()

        print(f"Sucessfully loaded the model to the memory")

        
        # Create pipeline
        self.pipe = pipeline(
            "translation",
            model = self.model,
            tokenizer = self.tokenizer,
            device = self.model.device
        )

    @torch.no_grad()
    def translate(self, sentence: str) -> str:
        text = f"{self.target_label}{sentence}"
        translation = self.pipe(text, num_beams = args.beams, max_length = args.max_new_tokens)
        return translation[0]["translation_text"]

    
    @torch.no_grad()
    def translate_batch(self, sentences: list[str]) -> list[str]:
        texts = [f"{self.target_label}{sentence}" for sentence in sentences]
        translations = self.pipe(texts, batch_size = self.args.batch_size, num_beams = self.args.beams, max_length = args.max_new_tokens)
        return [t["translation_text"] for t in translations]


class NLLBTranslator_HF(AbstractTranslator):
    def __init__(self, args):
        super().__init__(args)

        self.LANGUAGE_IDS = {
            "en": "eng_Latn",
            "nb": "nob_Latn",
            "nn": "nno_Latn",
        }

        model_path = NLLB_MODEL_IDS[self.args.nllb_model]  #f"facebook/nllb-200-3.3B"
        print(f"{model_path=}", flush = True)
        

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token = True,
            src_lang = self.LANGUAGE_IDS[self.args.slang],
            #padding_side = "left"
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            token = True,
        ).to(self.device).eval()

        print(f"Sucessfully loaded the model to the memory")


        # Create pipeline
        self.pipe = pipeline(
            "translation",
            model = self.model,
            tokenizer = self.tokenizer,
            src_lang = self.LANGUAGE_IDS[self.args.slang],
            tgt_lang = self.LANGUAGE_IDS[self.args.tlang],
            device = self.model.device
        )


    @torch.no_grad()
    def translate(self, sentence: str) -> str:
        translation = self.pipe(sentence, num_beams = args.beams)
        return translation[0]["translation_text"]

    
    @torch.no_grad()
    def translate_batch(self, sentences: list[str]) -> list[str]:
        translations = self.pipe(sentences, batch_size = self.args.batch_size, num_beams = self.args.beams, max_new_tokens = args.max_new_tokens)
        return [t["translation_text"] for t in translations]

class NLLBTranslator(AbstractTranslator):
    def __init__(self, args):
        super().__init__(args)

        # Add path to nllb branch of fairseq
        sys.path.insert(0, 'path/to/fairseq:nllb')
        from fairseq.models.transformer import TransformerModel
        from fairseq.data.data_utils import collate_tokens
        from fairseq import checkpoint_utils, options, tasks, utils
        
        self.LANGUAGE_IDS = {
            "en": "eng_Latn",
            "nb": "nob_Latn",
            "nn": "nno_Latn",
        }
        
        model_dir = Path(args.model_dir)
        model_path = model_dir / args.model
        spm_model = model_dir / args.spm
        dict = model_dir / args.dict

        input_args=[
            str(model_dir),
            '--path', str(model_path),
            '--task', 'translation_multi_simple_epoch',
            '--source-lang', self.LANGUAGE_IDS[args.slang],
            '--target-lang', self.LANGUAGE_IDS[args.tlang],
            '--bpe', 'sentencepiece',
            '--sentencepiece-model', str(spm_model),
            '--fixed-dictionary', str(dict),
            '--sacrebleu',
            '--remove-bpe', 'sentencepiece',
            '--beam', str(args.beams),
            '--encoder-langtok', args.encoder_langtok,
            '--decoder-langtok',
            '--langs', "ace_Arab,ace_Latn,acm_Arab,acq_Arab,aeb_Arab,afr_Latn,ajp_Arab,aka_Latn,amh_Ethi,apc_Arab,arb_Arab,ars_Arab,ary_Arab,arz_Arab,asm_Beng,ast_Latn,awa_Deva,ayr_Latn,azb_Arab,azj_Latn,bak_Cyrl,bam_Latn,ban_Latn,bel_Cyrl,bem_Latn,ben_Beng,bho_Deva,bjn_Arab,bjn_Latn,bod_Tibt,bos_Latn,bug_Latn,bul_Cyrl,cat_Latn,ceb_Latn,ces_Latn,cjk_Latn,ckb_Arab,crh_Latn,cym_Latn,dan_Latn,deu_Latn,dik_Latn,dyu_Latn,dzo_Tibt,ell_Grek,eng_Latn,epo_Latn,est_Latn,eus_Latn,ewe_Latn,fao_Latn,pes_Arab,fij_Latn,fin_Latn,fon_Latn,fra_Latn,fur_Latn,fuv_Latn,gla_Latn,gle_Latn,glg_Latn,grn_Latn,guj_Gujr,hat_Latn,hau_Latn,heb_Hebr,hin_Deva,hne_Deva,hrv_Latn,hun_Latn,hye_Armn,ibo_Latn,ilo_Latn,ind_Latn,isl_Latn,ita_Latn,jav_Latn,jpn_Jpan,kab_Latn,kac_Latn,kam_Latn,kan_Knda,kas_Arab,kas_Deva,kat_Geor,knc_Arab,knc_Latn,kaz_Cyrl,kbp_Latn,kea_Latn,khm_Khmr,kik_Latn,kin_Latn,kir_Cyrl,kmb_Latn,kon_Latn,kor_Hang,kmr_Latn,lao_Laoo,lvs_Latn,lij_Latn,lim_Latn,lin_Latn,lit_Latn,lmo_Latn,ltg_Latn,ltz_Latn,lua_Latn,lug_Latn,luo_Latn,lus_Latn,mag_Deva,mai_Deva,mal_Mlym,mar_Deva,min_Latn,mkd_Cyrl,plt_Latn,mlt_Latn,mni_Beng,khk_Cyrl,mos_Latn,mri_Latn,zsm_Latn,mya_Mymr,nld_Latn,nno_Latn,nob_Latn,npi_Deva,nso_Latn,nus_Latn,nya_Latn,oci_Latn,gaz_Latn,ory_Orya,pag_Latn,pan_Guru,pap_Latn,pol_Latn,por_Latn,prs_Arab,pbt_Arab,quy_Latn,ron_Latn,run_Latn,rus_Cyrl,sag_Latn,san_Deva,sat_Olck,scn_Latn,shn_Mymr,sin_Sinh,slk_Latn,slv_Latn,smo_Latn,sna_Latn,snd_Arab,som_Latn,sot_Latn,spa_Latn,als_Latn,srd_Latn,srp_Cyrl,ssw_Latn,sun_Latn,swe_Latn,swh_Latn,szl_Latn,tam_Taml,tat_Cyrl,tel_Telu,tgk_Cyrl,tgl_Latn,tha_Thai,tir_Ethi,taq_Latn,taq_Tfng,tpi_Latn,tsn_Latn,tso_Latn,tuk_Latn,tum_Latn,tur_Latn,twi_Latn,tzm_Tfng,uig_Arab,ukr_Cyrl,umb_Latn,urd_Arab,uzn_Latn,vec_Latn,vie_Latn,war_Latn,wol_Latn,xho_Latn,ydd_Hebr,yor_Latn,yue_Hant,zho_Hans,zho_Hant,zul_Latn",
            '--lang-pairs', f"{self.LANGUAGE_IDS[args.slang]}-{self.LANGUAGE_IDS[args.tlang]}",
            "--add-data-source-prefix-tags"
        ]

        if args.fp16:
            input_args.append("--fp16")


        if args.lenpen is not None:
            input_args.extend([
                "--lenpen", str(args.lenpen), #1.0
            ])
        if args.no_repeat_ngram_size is not None:
            input_args.extend([
                "--no-repeat-ngram-size", str(args.no_repeat_ngram_size), #3
            ])
        
        
        # Initialize the Fairseq parser and add arguments as done in fairseq-interactive
        parser = options.get_interactive_generation_parser()
        self.fairseq_args = options.parse_args_and_arch(parser, input_args = input_args)
        

        self.langtoken = f"__{self.LANGUAGE_IDS[args.tlang]}__"
        
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
        input_ids = self.task.source_dictionary.encode_line(tokenized_input, append_eos=False).long().unsqueeze(0)
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

        if detokenized_output.startswith(self.langtoken):
            # NLLB langtokens on the form '__nno_Latn__' use 12 characters
            detokenized_output = detokenized_output[13:]
    
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
                # NLLB langtokens on the form '__nno_Latn__' use 12 characters
                decoded_output = decoded_output[13:]
    
            results.append(decoded_output)
    
        return results


class AlmaTranslator(AbstractTranslator):
    def __init__(self, args):
        super().__init__(args)

        from peft import PeftModel

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
        input_length = model_inputs.input_ids.shape[1]
    

        if args.alma_sample:
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
            )

        else:
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=args.beams,
            )
        
        #return self.tokenizer.batch_decode(generated_ids[:, input_length:])
        # Default to splitting at new line
        return [sent.split("\n")[0].strip() for sent in self.tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens = True)]

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

    if args.method == "google":
        method_map = {
            "py": GoogleTranslatorPy,
            "deep" : GoogleTranslatorDeep,
            "api" : GoogleTranslatorAPI,
        }
        translator = method_map[args.google_method](args)
    elif args.method == "nllb":
        translator = NLLBTranslator(args)
    elif args.method == "nllb_hf":
        translator = NLLBTranslator_HF(args)
    elif args.method == "opus_hf":
        translator = OpusTranslator_HF(args)
    elif args.method == "alma":
        translator = AlmaTranslator(args)
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    if not args.no_eval:
        if args.output_dir is None:
            if args.method == "nllb_hf":
                model_string = f"_{args.nllb_model}"
            elif args.method == "opus_hf":
                model_string = f"_{args.opus_model}"
            elif args.method == "google":
                model_string = f"_{args.google_method}"
            else:
                model_string = ""
            output_dir = Path("NEW_TEST_SET") / f"{args.method}{model_string}__{args.slang}_{args.tlang}{args.output_suffix}"
        else:
            output_dir = Path("NEW_TEST_SET") / args.output_dir
        output_dir.mkdir()
        
        datasets, swap = get_datasets(args.slang, args.tlang)
        for dataset in datasets[args.start_dataset:]:
            with open(dataset, "r", encoding = "utf-8") as test_file:
                lines = test_file.read().strip().split("\n")
    
            output_file = output_dir / f"{dataset.name}.hyp"
            ref_file = output_dir / f"{dataset.name}.ref"            
            
            # All models with batch size 1
            if args.batch_size == 1:
                with open(output_file, "w", encoding = "utf-8") as hyp, \
                    open(ref_file, "w", encoding = "utf-8") as ref:

                    start_time = time.time()
                    
                    for line in tqdm(lines, mininterval = 5):
                        reference = line.split("\t")[0 if swap else 1].strip()
    
                        source_sent = line.split("\t")[1 if swap else 0].strip()
                        if args.google_method == "py" and args.method == "google":
                            hypothesis = asyncio.run(translator.translate(source_sent))
                        else:
                            hypothesis = translator.translate(source_sent)
                        
                        ref.write(f"{reference}\n")
                        hyp.write(f"{hypothesis}\n")

                    # Calculate elapsed time and print average time per sentence
                    elapsed_time = time.time() - start_time
                    average_time_per_sentence = elapsed_time / len(lines)
                    
                    # Write to file, should be ingored when computing metrics
                    hyp.write(f"##{average_time_per_sentence}\t{args.batch_size}\t{gpu_name}\t{gpu_vram}\n")
    
            #All methods with batch size above 1 except except normistralNshot and nort5
            elif args.batch_size > 1:
                with open(output_file, "w", encoding = "utf-8") as hyp, \
                    open(ref_file, "w", encoding = "utf-8") as ref:

                    # Time it
                    start_time = time.time()

                        
                    for i in tqdm(range(0, len(lines), args.batch_size), mininterval = 5):
                        references = [line.split("\t")[0 if swap else 1].strip() for line in lines[i:i+args.batch_size]]
                        
                        source_sents = [line.split("\t")[1 if swap else 0].strip() for line in lines[i:i+args.batch_size]]
                        if args.google_method == "py" and args.method == "google":
                            hypotheses = asyncio.run(translator.translate_batch(source_sents))
                        else:
                            hypotheses = translator.translate_batch(source_sents)
    
                        for r in references:
                            ref.write(f"{r}\n")
                        for h in hypotheses:
                            hyp.write(f"{h}\n")

                    # Calculate elapsed time and print average time per sentence
                    elapsed_time = time.time() - start_time
                    average_time_per_sentence = elapsed_time / len(lines)
                    
                    print(f"{dataset.name}: Average time per sentence: {average_time_per_sentence:.4f} seconds.")
                    # Write to file, should be ingored when computing metrics
                    hyp.write(f"##{average_time_per_sentence}\t{args.batch_size}\t{gpu_name}\t{gpu_vram}\n")

