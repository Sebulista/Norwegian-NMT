""" Translate from X -> Y in a two step fashion through an intermediate language Z,
such that model 1 translates from X -> Z and model 2 from Z -> Y"""
import random
import sys
import torch
import logging

from typing import Literal
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

import sentencepiece as spm
from fairseq.models.transformer import TransformerModel
from fairseq.data.data_utils import collate_tokens
from fairseq import checkpoint_utils, options, tasks, utils

from semantic_similarity import SimilarityScorer
from evaluation.evaluate import AbstractTranslator, NorT5Translator, NorMistralTranslator, NorMistralTranslatorNShot, FairseqMonoTranslator, FairseqMultiTranslator, get_gpu_vram_info, MBR_ENSEMBLE, MBR_MAP, init_mbr_metrics, init_mbr_metric, get_datasets


def parse_args():
    parser = ArgumentParser()
    # FIRST STEP
    parser.add_argument("--slang_1", type = str, choices = ["nb", "nn", "en"], help = "source language")
    parser.add_argument("--tlang_1", type = str, choices = ["nb", "nn", "en"], help = "target language")
    parser.add_argument("--method_1", type = str, help = "Translation method (model)")
    #Fairseq args
    parser.add_argument("--spm_1", type=str, default = "", help = "Path to sentencepiece model")
    parser.add_argument("--model_dir_1", type=str, default = "", help = "Path to model dir with dict")
    parser.add_argument("--model_1", type=str, default = "", help = "Path to fairseq checkpoint")
    #fairseq multi args
    parser.add_argument("--encoder_langtok_1", type = str, choices = ["tgt", "src"], help = "Whether to use source side or target side encoder langtoken")
    parser.add_argument("--decoder_langtok_1", action = "store_true", help = "Whether to add a decoder langtoken")
    parser.add_argument("--tagged_bt_1", action = "store_true", help = "Whether to add tagged backtranslation tokens for compatability.")

    
    # SECOND STEP
    parser.add_argument("--slang_2", type = str, choices = ["nb", "nn", "en"], help = "source language")
    parser.add_argument("--tlang_2", type = str, choices = ["nb", "nn", "en"], help = "target language")
    parser.add_argument("--method_2", type = str, help = "Translation method (model)")
    #Fairseq args
    parser.add_argument("--spm_2", type=str, default = "", help = "Path to sentencepiece model")
    parser.add_argument("--model_dir_2", type=str, default = "", help = "Path to model dir with dict")
    parser.add_argument("--model_2", type=str, default = "", help = "Path to fairseq checkpoint")
    #fairseq multi args
    parser.add_argument("--encoder_langtok_2", type = str, choices = ["tgt", "src"], help = "Whether to use source side or target side encoder langtoken")
    parser.add_argument("--decoder_langtok_2", action = "store_true", help = "Whether to add a decoder langtoken")
    parser.add_argument("--tagged_bt_2", action = "store_true", help = "Whether to add tagged backtranslation tokens for compatability.")

    
    # SHARED
    parser.add_argument("--batch_size", type = int, default = 1, help = "Batch size")
    parser.add_argument("--output_dir", type = str, default = None, help = "Name of output directory")
    parser.add_argument("--output_suffix", type = str, default = "", help = "Additional suffix to the output directory")
    parser.add_argument("--seed", type = int, default = 73, help = "Random seed")
    parser.add_argument("--beams", type = int, default = 5, help = "Number of beams for beam search")
    parser.add_argument("--n_hyp", type = int, default = 5, help = "Number of beams")

    #Generate synthetic data args (NOT USED IN THIS FILE)
    parser.add_argument("--dataset", type = str, default = None, help = "Path to dataset to load, only used in generate_synthetic_data.py")
    parser.add_argument("--scorer", choices = ["top", "sim", "both"], default = "sim", help = "Which scorer to use. 'sim': LABSE similarity scoring, 'top': naive choose top, 'both': create two files, one top and one sim")

    # FOR COMPATABILITY, NOT IMPLEMENTED
    parser.add_argument("--sample_epsilon", type = float, default = None, help = "If provided, epsilon sampling is used")
    parser.add_argument("--mbr", type = str, default = None, choices = MBR_MAP.keys(), help = "If provided, use the specifided utility for minimum bayes risk decoding")
    parser.add_argument("--mbr_ensemble", type = str, default = None, choices = MBR_ENSEMBLE.keys(), help = "Which MBR ensembling technique to use. Won't be used if not provided.")

    args = parser.parse_args()

    # Set do_mbr to True if either of the mbr arguments are provided
    args.do_mbr = (args.mbr is not None) or (args.mbr_ensemble is not None)
    
    return args


class TwoStepTranslator(AbstractTranslator):
    def __init__(self, args):
        super().__init__(args)

        MODEL_MAP = {
            "nort5": NorT5Translator,
            "normistral": NorMistralTranslator,
            "normistralNshot": NorMistralTranslatorNShot,
            "fairseq_mono": FairseqMonoTranslator,
            "fairseq_multi": FairseqMultiTranslator,
        }

        try:
            translator_class_1 = MODEL_MAP[args.method_1]
            translator_class_2 = MODEL_MAP[args.method_2]

        except KeyError as e:
            raise ValueError(f"Unsupported method: {e.args[0]}")

        # CHECKS
        assert args.tlang_1 == args.slang_2, f"Two step translation is not compatible, {args.tlang_1} != {args.slang_2}"
        
        # FAIRSEQ
        # TRANSLATOR 1
        args.model_dir = args.model_dir_1
        args.model = args.model_1
        args.spm = args.spm_1
        args.slang = args.slang_1
        args.tlang = args.tlang_1
        args.encoder_langtok = args.encoder_langtok_1
        args.decoder_langtok = args.decoder_langtok_1
        args.tagged_bt = args.tagged_bt_1
        self.translator_1 = translator_class_1(args)
        print(f"LOADED {args.slang}_{args.tlang} {args.method_1} MODEL ({type(self.translator_1)})")
        
        # TRANSLATOR 2
        args.model_dir = args.model_dir_2
        args.model = args.model_2
        args.spm = args.spm_2
        args.slang = args.slang_2
        args.tlang = args.tlang_2
        args.encoder_langtok = args.encoder_langtok_2
        args.decoder_langtok = args.decoder_langtok_2
        args.tagged_bt = args.tagged_bt_2
        self.translator_2 = translator_class_2(args)
        print(f"LOADED {args.slang}_{args.tlang} {args.method_2} MODEL ({type(self.translator_2)})")

    
    @torch.no_grad()
    def translate(self, sentence: str) -> str:
        hypothesis_1 = self.translator_1.translate(sentence)
        hypothesis_2 = self.translator_2.translate(hypothesis_1)

        return hypothesis_2

    @torch.no_grad()
    def translate_batch(self, sentences: list[str]) -> list[str]:
        hypotheses_1 = self.translator_1.translate_batch(sentences)
        hypotheses_2 = self.translator_2.translate_batch(hypotheses_1)

        return hypotheses_2




if __name__ == "__main__":
    random.seed(73)
    get_gpu_vram_info()
    print(f"\n{' '.join(sys.argv)}\n")
    args = parse_args()

    translator = TwoStepTranslator(args)

    if args.output_dir is None:        
        output_dir = Path(".") / f"two_step_{args.method_1}_{args.method_2}_{args.slang_1}_{args.tlang_2}{args.output_suffix}"
    else:
        output_dir = Path(".") /  args.output_dir
    output_dir.mkdir()
    
    datasets, swap = get_datasets(args.slang_1, args.tlang_2)
    for dataset in datasets:
        with open(dataset, "r", encoding = "utf-8") as test_file:
            lines = test_file.read().strip().split("\n")

        output_file = output_dir / f"{dataset.name}.hyp"
        ref_file = output_dir / f"{dataset.name}.ref"

        # All models with batch size 1
        if args.batch_size == 1:
            with open(output_file, "w", encoding = "utf-8") as hyp, \
                open(ref_file, "w", encoding = "utf-8") as ref:
                for line in tqdm(lines):
                    reference = line.split("\t")[0 if swap else 1].strip()

                    source_sent = line.split("\t")[1 if swap else 0].strip()
                    hypothesis = translator.translate(source_sent)
                    
                    ref.write(f"{reference}\n")
                    hyp.write(f"{hypothesis}\n")


        else:
            with open(output_file, "w", encoding = "utf-8") as hyp, \
                open(ref_file, "w", encoding = "utf-8") as ref:
                for i in tqdm(range(0, len(lines), args.batch_size)):
                    references = [line.split("\t")[0 if swap else 1].strip() for line in lines[i:i+args.batch_size]]
                    
                    source_sents = [line.split("\t")[1 if swap else 0].strip() for line in lines[i:i+args.batch_size]]
                    
                    hypotheses = translator.translate_batch(source_sents)
     
                    for r in references:
                        ref.write(f"{r}\n")
                    for h in hypotheses:
                        hyp.write(f"{h}\n")
