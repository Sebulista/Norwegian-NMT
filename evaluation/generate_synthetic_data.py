""" Program to generate synthetic data and filter with cosine similarity computed by LaBSE embeddings """

import random
import sys
import torch
import logging

from typing import Literal
from pathlib import Path
from tqdm import tqdm

import sentencepiece as spm
from fairseq.models.transformer import TransformerModel
from fairseq.data.data_utils import collate_tokens
from fairseq import checkpoint_utils, options, tasks, utils

from semantic_similarity import SimilarityScorer
from evaluation.evaluate import AbstractTranslator, NorT5Translator, NorMistralTranslator, NorMistralTranslatorNShot, get_gpu_vram_info, MBR_ENSEMBLE, MBR_MAP, init_mbr_metrics, init_mbr_metric, parse_args




class FairseqMonoTranslator(AbstractTranslator):
    def __init__(self, args):
        super().__init__(args)

        model_dir = Path(args.model_dir)
        model_path = model_dir / args.model
        spm_model = model_dir / args.spm

        print(f"Decoding strategy: {'MBR' if args.do_mbr else 'BEAM SEARCH'}")

        input_args = [
            str(model_dir),
            '--path', str(model_path),
            '--user-dir', '../custom_fairseq',
            '--source-lang', args.slang,
            '--target-lang', args.tlang,
            '--bpe', 'sentencepiece',
            '--sentencepiece-model', str(spm_model),
            '--sacrebleu',
            '--remove-bpe', 'sentencepiece',
            '--beam', str(args.n_hyp),
            '--nbest', str(args.n_hyp),
            '--fp16',
            '--seed', str(args.seed),
        ]
        
        # Initialize the Fairseq parser and add arguments as done in fairseq-interactive
        parser = options.get_interactive_generation_parser()
        if args.do_mbr:
            self.fairseq_args = options.parse_args_and_arch(parser, input_args = input_args + [
                '--task', 'epsilon_translation',
                '--sampling',
                '--sampling-epsilon', str(0.02),
            ])

        else:
            self.fairseq_args = options.parse_args_and_arch(parser, input_args = input_args + [
                '--task', 'translation',
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

        # Score using MBR
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
    def translate(self, sentence: str) -> list[str]:
        # Tokenize input text using SentencePiece
        tokenized_input = self.bpe.encode(sentence)
    
        # Convert to tensor and predict
        input_ids = self.task.src_dict.encode_line(tokenized_input, append_eos=True).long().unsqueeze(0).to(self.device)
        lengths = torch.LongTensor([input_ids.size(1)]).to(self.device)
        
        sample = {'net_input': {'src_tokens': input_ids, 'src_lengths': lengths}}
    
        # Perform the translation
        translations = self.task.inference_step(self.generator, self.models, sample)

        translation_tokens = [t['tokens'] for t in translations[0]]
        
        # Convert tokens to text
        decoded_output = [self.task.tgt_dict.string(t) for t in translation_tokens]
        # Detokenize output using SentencePiece
        hypotheses = [self.bpe.decode(d) for d in decoded_output]

        if args.do_mbr:
            mbr_decoded_output = self.decoder.decode(hypotheses, hypotheses, source=sentence, nbest=1)
            
            return mbr_decoded_output.sentence
        # Beam search
        else:
            return hypotheses
            
    @torch.no_grad()
    def translate_batch(self, sentences: list[str]) -> list[list[str]]:
        """ Translate a batch of sentences such that for each source sentence,
        args.n_hyp hypotheses are produced
        
        Arguments:
            sentences (list[str]): A list of args.batch_size sentences

        Returns:
            A list of string lists of dimension (args.batch_size, args.n_hyp))
        """
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
            output_tokens = [t['tokens'] for t in translation]
            hypotheses = [self.bpe.decode(self.task.tgt_dict.string(t)) for t in output_tokens]

            if args.do_mbr:
                mbr_decoded_output = self.decoder.decode(hypotheses, hypotheses, source=sentences[i], nbest=1)
                
                results.append(mbr_decoded_output.sentence[0])
            else:
                results.append(hypotheses)
    
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
            '--beam', str(args.n_hyp),
            '--nbest', str(args.n_hyp),
            '--lang-pairs', "nb-nn,nn-nb",
            '--seed', str(args.seed),
            '--fp16',
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
        
        
        for model in self.models:
            model.to(self.device).eval()
        
        # Initialize the BPE process
        self.bpe = self.task.build_bpe(self.fairseq_args)
        
        # Build generator
        self.generator = self.task.build_generator(self.models, self.fairseq_args)

    @torch.no_grad()
    def translate(self, sentence: str) -> list[str]:
        """Given one input sentence, output a list nbest of hypotheses"""
        # Tokenize input text using SentencePiece
        tokenized_input = self.bpe.encode(sentence)
    
        # Convert to tensor and predict
        input_ids = self.task.source_dictionary.encode_line(tokenized_input, append_eos=True).long().unsqueeze(0).to(self.device)
        lengths = torch.LongTensor([input_ids.size(1)]).to(self.device)
        
        sample = {'net_input': {'src_tokens': input_ids, 'src_lengths': lengths}}
    
        # Perform the translation
        translations = self.task.inference_step(self.generator, self.models, sample)
        
        translation_tokens = [t['tokens'] for t in translations[0]]
        
        # Convert tokens to text
        decoded_output = [self.task.tgt_dict.string(t) for t in translation_tokens]
    
        # Detokenize output using SentencePiece
        hypotheses = [self.bpe.decode(d) for d in decoded_output]
    
        return hypotheses

    @torch.no_grad()
    def translate_batch(self, sentences: list[str]) -> list[list[str]]:
        """ Translate a batch of sentences such that for each source sentence,
        args.n_hyp hypotheses are produced

        Arguments:
            sentences (list[str]): A list of args.batch_size sentences

        Returns:
            A list of string lists of dimension (args.batch_size, args.n_hyp))
        """
        # Tokenizing input sentences and creating batch
        tokens_list = [
            self.task.source_dictionary.encode_line(
                line=self.bpe.encode(sentence),
                add_if_not_exist=False,
                append_eos=True
            ).long()
            for sentence in sentences
        ]
        
        src_lengths = torch.LongTensor([tokens.size(0) for tokens in tokens_list]).to(self.device)
        src_tokens = collate_tokens(
            tokens_list, pad_idx=self.task.source_dictionary.pad(), eos_idx=self.task.source_dictionary.eos()
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
            output_tokens = [t['tokens'] for t in translation]
            hypotheses = [self.bpe.decode(self.task.target_dictionary.string(t)) for t in output_tokens]

            for i in range(len(hypotheses)):
                if hypotheses[i].startswith(self.langtoken):
                    hypotheses[i] = hypotheses[i][7:]
    
            results.append(hypotheses)
    
        return results


class TopScorer:
    def __init__(self, top_k = 1):
        self.top_k = top_k
    
    def filter(self, src: str, tgts: list[str]) -> list[str]:
        """Filter hypotheses by naively outputting the top_k targets"""

        return tgts[:self.top_k]

    
    def filter_batch(self, sources: list[str], targets: list[list[str]]) -> list[list[str]]:
        """Filter hypothese by naively outputting the top_k targets of each batch etnry"""

        filtered = [tgts[:self.top_k] for tgts in targets]
        
        return filtered


def format_batch(sources: list[str], filtered_hypotheses: list[list[str]]) -> list[str]:
    """Format the filtered batch for writing as tab separated fields
    
    Arguments:
        sources (list[str]): A list of N sources
        filtered_hypotheses (list[list[str]]): A list containg string lists of dimension (N, irange(0, top_k))
        
    Returns:
        A list of formatted tab separated strings
    """

    formatted_lines = []
    for source, hypotheses in zip(sources, filtered_hypotheses):
        if not hypotheses:
            continue
        for hyp in hypotheses:
            formatted_line = f"{source}\t{hyp}\n"
            formatted_lines.append(formatted_line)

    return formatted_lines


def get_dataset(slang: Literal["nb", "en", "nn"]) -> str:
    """Return the monolingual dataset given the target language"""
    source_folder = Path("/cluster/work/projects/ec30/ec-jonassf/data/mono/data")
    for child in source_folder.iterdir():
        if child.is_file():
            if child.name.split(".")[-1] == slang:
                return child
    return None


def init_logger():
    logger = logging.getLogger(__name__)
    #logger.handlers = []  # Removing default handler to avoid duplication of log messages
    logging.getLogger().handlers.clear()
    logger.setLevel(logging.DEBUG)

    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(h)

    return logger


if __name__ == "__main__":
    random.seed(73)
    get_gpu_vram_info()

    logger = init_logger()
    
    print(f"\n{' '.join(sys.argv)}\n")
    args = parse_args()
    logger.info("Arguments processed.")

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

    logger.info(f"Using method: {args.method} ({type(translator)})")

    if args.scorer == "top":
        scorer = TopScorer()
        logger.info(f"Using scorer: {args.scorer} ({type(scorer)})")
    elif args.scorer == "both":
        scorer_sim = SimilarityScorer(translator.device, args.batch_size)
        scorer_top = TopScorer()
        logger.info(f"Using both scorers: ({type(scorer_sim)}, {type(scorer_top)})")
    else:
        scorer = SimilarityScorer(translator.device, args.batch_size)
        logger.info(f"Using scorer: {args.scorer} ({type(scorer)})")
    
    if args.output_dir is None:
        output_dir = Path("SET/OUTPUT_FOLDER") / f"{args.method}{args.output_suffix}"
    else:
        output_dir = Path("SET/OUTPUT_FOLDER") / args.output_dir
    output_dir.mkdir(exist_ok = True)

    logger.debug(f"Output directory: {output_dir}")

    if args.scorer == "both":
        output_file_sim = output_dir / f"{args.method}_sim{args.output_suffix}.{args.slang}-{args.tlang}"
        output_file_top = output_dir / f"{args.method}_top{args.output_suffix}.{args.slang}-{args.tlang}"
        assert not output_file_sim.is_file()
        assert not output_file_top.is_file()
        logger.debug(f"Using two scorers (sim and top) and outputting two files:\n{output_file_sim}\n{output_file_top}")
    else:
        output_file = output_dir / f"{args.method}{args.output_suffix}.{args.slang}-{args.tlang}"
        assert not output_file.is_file()
        logger.debug(f"Output file: {output_file.name}")
    
    if args.dataset is not None:
        dataset = Path(args.dataset)
        assert dataset.is_file()
    else:
        dataset = get_dataset(args.slang)
    logger.info(f"LOADED {dataset.name}")
    with open(dataset, "r", encoding = "utf-8") as in_file:
        lines = in_file.read().strip().split("\n")
        logger.info(F"LINES IN DATASET: {len(lines):,}")


        if args.scorer == "both":
            with open(output_file_sim, "w", encoding = "utf-8") as out_sim, \
                open(output_file_top, "w", encoding = "utf-8") as out_top:
                for i in tqdm(range(0, len(lines), args.batch_size), mininterval=60):
                    source_sents = [line.strip() for line in lines[i:i+args.batch_size]]
                    hypotheses = translator.translate_batch(source_sents)
                    # Filter
                    filtered_sim_hypotheses = scorer_sim.filter_batch(source_sents, hypotheses)
                    filtered_top_hypotheses = scorer_top.filter_batch(source_sents, hypotheses)
                    
                    filtered_sim_pairs = format_batch(source_sents, filtered_sim_hypotheses)
                    filtered_top_pairs = format_batch(source_sents, filtered_top_hypotheses)
                    for formatted_line in filtered_sim_pairs:
                        out_sim.write(formatted_line)
                    for formatted_line in filtered_top_pairs:
                        out_top.write(formatted_line)
        
        else:
            with open(output_file, "w", encoding = "utf-8") as out_file:
                for i in tqdm(range(0, len(lines), args.batch_size), mininterval=60):
                    source_sents = [line.strip() for line in lines[i:i+args.batch_size]]
                    hypotheses = translator.translate_batch(source_sents)
                    # Filter
                    filtered_hypotheses = scorer.filter_batch(source_sents, hypotheses)
                    filtered_pairs = format_batch(source_sents, filtered_hypotheses)
                    for formatted_line in filtered_pairs:
                        out_file.write(formatted_line)

