""" Utility program to compute metrics specified by command line arguments given a folder of system translations and their references """
import os

# Tensorflow
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Set TOKENIZERS_PARALLELISM so it doesn't annoy us
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torchmetrics
from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
import regex as re
import numpy as np


from sacrebleu.metrics import BLEU, CHRF
from comet import download_model, load_from_checkpoint

from utils import format_kiwi, format_comet, format_metricx, format_metricx_qe
from metricx.metricx24.score import MetricX

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", message="The `srun` command is available on your system but is not used.")


DATASETS = {
    "tatoeba": "Tatoeba-v2023-04-12",
    "flores_dev": "flores_dev",
}

GPUS = {
    "NVIDIA A100 80GB PCIe": "A100",
    "NVIDIA A100-PCIE-40GB": "A100-40GB",
    "NVIDIA A40": "A40",
    "NVIDIA GeForce RTX 3090": "3090",
    "NVIDIA L40S": "L40S",
    "CPU": "CPU",
}

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("dir", type = str, help = "Directory with ref and hyp files")
    parser.add_argument("--bert_score", "-b", action = "store_true", help = "Whether or not to calculate bert score")
    parser.add_argument("--comet", action = "store_true", help = "Whether or not to calculate comet (reference-based metric)")
    parser.add_argument("--kiwi", action = "store_true", help = "Whether or not to calculate comet kiwi score (quality estimation metric)")
    parser.add_argument("--metricx", action = "store_true", help = "Whether or not to calculate metricx score (reference-based)")
    parser.add_argument("--metricx_qe", action = "store_true", help = "Whether or not to calculate QE metricx score")
    parser.add_argument("--model", "-m", type=str, default = None, help = "If specified, only this model directory will be used")
    parser.add_argument("--glob", "-g", type = str, default = None, help = "Glob expression for files to process")
    parser.add_argument("--sorted", "-s", action = "store_true", help = "If provided, output model evaluations in sorted order within each direction")
    parser.add_argument("--clean", "-c", action = "store_true", help = "Whether or not to clean up trailing repetitive punctuation")
    parser.add_argument("--clean_EOS", action = "store_true", help = "Whether or not to split on '<' to remove part after EOS token")
    parser.add_argument("--batch_size", type = int, default = 64, help = "Batch size of metric calculation")
    
    parser.add_argument("--dataset", type = str, default = None, choices = DATASETS.keys(), help = "If used, only perform evaluation on this dataset.")

    args = parser.parse_args()
    return args


def get_slang_tlang(folder_name):
    pat = re.compile(r"(nb|en|nn)_(nb|en|nn)")
    match = re.search(pat, folder_name)
    if match is not None:
        return match.group()
    return ""

def clean_punctuation(sents: list[str]):
    new_sents = [re.sub(r"([\s\.\,\-\_\"]){10,}", lambda m: m.group(0)[0], sent) for sent in sents]
    return new_sents


class ScoreObject():
    """Class to store corpus scores for each test set"""
    def __init__(self, folder: Path, bleu_scores: dict[str, float], chrf_scores: dict[str, float], chrfPP_scores: dict[str, float], bert_scores: dict[str, float], comet_scores: dict[str, float], kiwi_scores: dict[str, float], metricx_scores: dict[str, float], metricx_qe_scores: dict[str, float], runtimes: dict[str, str]):
        self.folder = folder
        # Add macro average to the end of each line
        self.bleu_scores = list(bleu_scores.values())+[np.mean(list(bleu_scores.values()))]
        self.chrf_scores = list(chrf_scores.values())+[np.mean(list(chrf_scores.values()))]
        self.chrfPP_scores = list(chrfPP_scores.values())+[np.mean(list(chrfPP_scores.values()))]
        self.bert_scores = list(bert_scores.values())+[np.mean(list(bert_scores.values()))] if bert_scores else []
        self.comet_scores = list(comet_scores.values())+[np.mean(list(comet_scores.values()))] if comet_scores else []
        self.kiwi_scores = list(kiwi_scores.values())+[np.mean(list(kiwi_scores.values()))] if kiwi_scores else []
        self.metricx_scores = list(metricx_scores.values())+[np.mean(list(metricx_scores.values()))] if metricx_scores else []
        self.metricx_qe_scores = list(metricx_qe_scores.values())+[np.mean(list(metricx_qe_scores.values()))] if metricx_qe_scores else []

        self.runtimes = self.get_runtimes(runtimes) if runtimes else []

    def get_runtimes(self, runtimes: dict[str, str]):
        throughputs = []
        for name, runstring in runtimes.items():
            T, bsz, gpu, vram = runstring[2:].strip().split("\t")
            throughput = 1/float(T)
            throughputs.append(throughput)
            #runtime_string = f"{throughput:.2f}@{bsz}@{GPUS[gpu]}"

        avg_T = sum(throughputs) / len(throughputs)
        sum_string = f"{avg_T:.2f}@{bsz}@{GPUS[gpu]}"

        return throughputs + [sum_string]

    def get_cmp_value(self):
        """Return the compare value, currently average BLEU score"""
        return self.bleu_scores[-1]


class ScoreManager():
    """Class to manage several score objects, sort and print these"""
    def __init__(self):
        self.scores = []

    def add_score(self, score: ScoreObject):
        self.scores.append(score)

    def empty(self):
        self.scores.clear()

    def sort_scores(self, ascending = False):
        def sort_key(score_obj: ScoreObject):
            macro_bleu = score_obj.get_cmp_value()
            return macro_bleu

        self.scores = sorted(self.scores, key = sort_key, reverse = not ascending)

    def print_score(self, score: ScoreObject):
        print(f"{score.folder.name}, BLEU, ", end = "")
        # Add macro average to the end of each line
        print(', '.join(f"{value:.2f}" for value in score.bleu_scores), end="")
        print("\n, CHRF, ", end = "")
        print(', '.join(f"{value:.2f}" for value in score.chrf_scores), end = "")
        print("\n, CHRF++, ", end = "")
        print(', '.join(f"{value:.2f}" for value in score.chrfPP_scores), end = "")
        
        if score.bert_scores:
            print("\n, BERTSCORE, ", end = "")
            print(', '.join(f"{value:.2%}" for value in score.bert_scores), end = "")

        if score.comet_scores:
            print("\n, Comet, ", end = "")
            print(', '.join(f"{value:.2%}" for value in score.comet_scores), end = "")

        if score.kiwi_scores:
            print("\n, CometKiwi, ", end = "")
            print(', '.join(f"{value:.2%}" for value in score.kiwi_scores), end = "")

        if score.metricx_scores:
            print("\n, MetricX, ", end = "")
            print(', '.join(f"{value:.2f}" for value in score.metricx_scores), end = "")

        if score.metricx_qe_scores:
            print("\n, MetricX QE, ", end = "")
            print(', '.join(f"{value:.2f}" for value in score.metricx_qe_scores), end = "")

        if score.runtimes:
            print("\n, Runtimes, ", end = "")
            print(', '.join(f"{value:.2f}" for value in score.runtimes[:-1]), end = "")
            print(f", {score.runtimes[-1]}", end = "")
        
        print()
        print()

    def print_last(self):
        self.print_score(self.scores[-1])

    def print_all(self):
        self.sort_scores()
        for score in self.scores:
            self.print_score(score)


def get_dataset(slang: str, tlang: str, name: str):
    """Get relevant test set"""
    
    if name.startswith("Tatoeba"):
        subfolder = "tatoeba"
    elif name.startswith("ntrex"):
        subfolder = "ntrex"
    elif name.startswith("flores"):
        subfolder = "flores"
    else:
        subfolder = "giellatekno"
        
    source_path = Path(f"SET/PATH/{subfolder}/")

    swap = False
    
    for file in source_path.iterdir():
        if file.name.split(".")[0].strip() == name.strip():
            s, t = file.name.split(".")[-1].split("-")
            #Correct way
            if s == slang and t == tlang:
                return file, swap
            elif s == tlang and t == slang:
                swap = True
                return file, swap


def get_gpu_vram_info():
    import torch
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


def main():
    """Loop through multiple model outputs, themselves with multiple test set outputs, and print out the score for each model"""
    get_gpu_vram_info()
    args = parse_args()
    if args.dataset is not None:
        skip = []
    else:
        skip = ["flores_dev"]

    bleu = BLEU()
    chrf = CHRF()
    chrfPP = CHRF(word_order=2)

    # Loading all neural metric models at once is inefficient, but should be no VRAM 
    # concern for inference
    if args.bert_score:
        import bert_score
        
        bert_scorer = bert_score.BERTScorer(model_type = "path/to/model", num_layers = 9)

    if args.comet:
        comet_scorer = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da", saving_directory = "path/to/models", local_files_only = True))
        comet_scorer.eval()

    if args.kiwi:
        kiwi_scorer = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da", saving_directory = "path/to/models", local_files_only = True))
        kiwi_scorer.eval()

    if args.metricx or args.metricx_qe:
        metricx_scorer = MetricX(args.batch_size)

    # Get folders and sort by lang direction
    if args.model is not None:
        folders = [Path(args.dir) / args.model]
        assert folders[0].is_dir()
    elif args.glob is not None:
        folders = [folder for folder in Path(args.dir).glob(args.glob) if folder.is_dir() and folder.name[0] not in [".", "_"]]
    else:
        folders = [folder for folder in Path(args.dir).iterdir() if folder.is_dir() and folder.name[0] not in [".", "_"]]
    sorted_folders = sorted(folders, key = lambda f: get_slang_tlang(f.name))

    # Debug print statement
    for f in sorted_folders:
        print(f.name, flush=True)

    current_direction = ""

    score_manager = ScoreManager()
    
    for i, folder in enumerate(sorted_folders):
        #Get all hypothesis and reference files
        if args.dataset is not None:
            hyps = sorted(list(folder.glob(f"{DATASETS[args.dataset]}.*.hyp")))
            refs = sorted(list(folder.glob(f"{DATASETS[args.dataset]}.*.ref")))
        else:
            hyps = sorted(list(folder.glob("*.hyp")))
            refs = sorted(list(folder.glob("*.ref")))

        # New language direction
        if get_slang_tlang(folder.name) != current_direction:
            
            if args.sorted and i != 0:
                score_manager.print_all()

            score_manager.empty()

            # Print current direction once
            current_direction = get_slang_tlang(folder.name)
            new_direction = True
            print(f"{'-'*30}{current_direction.upper()}{'-'*30}")
            # Print testset names
            print(f"{current_direction}, , ", end = "")
            print(", ".join([f.name.split(".")[0] for f in hyps if f.name.split(".")[0] not in skip]), end = "")
            print(", MACRO\n")

    
        bleu_scores, bert_scores, chrf_scores, chrfPP_scores, comet_scores, kiwi_scores, metricx_scores, metricx_qe_scores = {}, {}, {}, {}, {}, {}, {}, {}
        runtimes = {}
        
        for hyp, ref in zip(hyps, refs):
            # Get dataset name without extension
            dataset = hyp.name.split(".")[0]

            # Skip flores_dev as it is being used as dev set
            if dataset in skip:
                continue

            
            # Hypotheses
            with open(hyp, "r", encoding = "utf-8") as hyp_file:
                # The model might produce an empty translation, strip could therefore remove 'valuable' lines
                #hypotheses = hyp_file.read().strip().split("\n")
                hypotheses = hyp_file.read().rstrip().split("\n")
                if hypotheses[-1].startswith("##"):
                    runtimes[dataset] = hypotheses[-1]
                hypotheses = [line for line in hypotheses if not line.startswith("##")]
                if args.clean_EOS:
                    hypotheses = [hyp.split('<')[0] for hyp in hypotheses]
                if args.clean:
                    hypotheses = clean_punctuation(hypotheses)

            # References
            with open(ref, "r", encoding = "utf-8") as ref_file:
                references = ref_file.read().strip().split("\n")

                #if args.clean:
                #    references = clean_punctuation(references)

            # Sources
            if args.comet or args.kiwi or args.metricx or args.metricx_qe:
                slang, tlang = get_slang_tlang(folder.name).split("_")
                test_set, swap = get_dataset(slang, tlang, dataset)
                with open(test_set, "r", encoding = "utf-8") as source_file:
                    sources = [l.split("\t")[1 if swap else 0].strip() for l in source_file.read().strip().split("\n")]

            # Check if the hyp and ref file have the same length
            if len(hypotheses) != len(references):
                print(f"WARNING: Mismatch in the number of lines of the .hyp ({len(hypotheses)}) and .ref ({len(references)}) file for {dataset} ({folder})")
                continue

            #Calculate lexical metrics
            bleu_score = bleu.corpus_score(hypotheses, [references]).score
            chrf_score = chrf.corpus_score(hypotheses, [references]).score
            chrfPP_score = chrfPP.corpus_score(hypotheses, [references]).score

            # Neural metrics
            
            if args.bert_score:
                try:
                    all_preds = bert_scorer.score(hypotheses, references)
            
                    avg_scores = [s.mean(dim=0) for s in all_preds]
                    P = avg_scores[0].cpu().item()
                    R = avg_scores[1].cpu().item()
                    F1 = avg_scores[2].cpu().item()
                    #msg = f" P: {P:.6f} R: {R:.6f} F1: {F1:.6f}"
                    #print(msg)
                    bert_scores[dataset] = F1
                except RuntimeError:
                    print(f"FOLDER: {folder}")
                    print(f"DATSET: {dataset}")

            if args.comet:
                score_data = format_comet(sources, hypotheses, references)
                system_score = comet_scorer.predict(score_data, batch_size = args.batch_size, gpus = 1, progress_bar = False)["system_score"]
                
                comet_scores[dataset] = system_score

            if args.kiwi:
                score_data = format_kiwi(sources, hypotheses)
                system_score = kiwi_scorer.predict(score_data, batch_size = args.batch_size, gpus = 1, progress_bar = False)["system_score"]
                
                kiwi_scores[dataset] = system_score

            if args.metricx:
                score_data = format_metricx(sources, hypotheses, references)
                system_score = metricx_scorer.score(score_data, is_qe = False)

                metricx_scores[dataset] = system_score

            if args.metricx_qe:
                score_data = format_metricx_qe(sources, hypotheses)
                system_score = metricx_scorer.score(score_data, is_qe = True)

                metricx_qe_scores[dataset] = system_score
            
            bleu_scores[dataset] = bleu_score
            chrf_scores[dataset] = chrf_score
            chrfPP_scores[dataset] = chrfPP_score


        score = ScoreObject(folder, bleu_scores, chrf_scores, chrfPP_scores, bert_scores, comet_scores, kiwi_scores, metricx_scores, metricx_qe_scores, runtimes=runtimes)
        score_manager.add_score(score)
        
        if not args.sorted:
            score_manager.print_last()    

        # Print last folder when using sorted
        if args.sorted and i == len(sorted_folders) - 1:
            score_manager.print_all()

if __name__ == "__main__":
    main()
