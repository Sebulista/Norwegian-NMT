""" Utilities to calculate metric evaluation for fairseq models across all checkpoints on the validation sets.
This functionality is not provided with the fairseq task 'translation_multi_simple_epoch'"""
import os
import concurrent.futures
# Tensorflow
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Set TOKENIZERS_PARALLELISM so it doesn't annoy us
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#import torchmetrics
from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
import regex as re
import numpy as np
from typing import Literal
from tqdm import tqdm
import pickle

from sacrebleu.metrics import BLEU, CHRF

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", message="The `srun` command is available on your system but is not used.")


COL1_WIDTH = 20
COL2_WIDTH = 10

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("dir", type = str, help = "Directory with ref and hyp files")
    parser.add_argument("--multi", "-m", action = "store_true", help = "If provided, positional agument dir is a folder with folders to evaluate")
    parser.add_argument("--verbose", "-v", action = "store_true", help = "Print metrics for each checkpoint")
    parser.add_argument("--pickle", action = "store_true", help = "Save computed metrics as a pickle file")
    parser.add_argument("--parallel", "-P", action = "store_true")

    args = parser.parse_args()
    return args


def get_slang_tlang(folder_name):
    src, tgt = folder_name.split("__")[-1].split("_")
    return src, tgt


class ScoreObject():
    """Class to store metric scores for the validation set"""
    def __init__(self, folder: Path, bleu_score: float, chrf_score: float):
        self.folder = folder
        # Add macro average to the end of each line
        self.bleu_score = bleu_score
        self.chrf_score = chrf_score

    def get_cmp_value(self, metric: Literal["bleu", "chrf"]):
        """Return the compare value, currently average BLEU score"""
        return self.bleu_score if metric == "bleu" else self.chrf_score

class ModelManager():
    """Class to store the scores of each checkpoint within a model"""
    def __init__(self, model: Path):
        self.scores = []
        self.model = model

    def add_score(self, score: ScoreObject):
        self.scores.append(score)

    def sort_literal(self):
        return sorted(self.scores, key = lambda x: int(x.folder.name.split(".")[0][10:]))

    def sort_scores(self, metric: Literal["bleu", "chrf"]):
        def sort_key(score_obj: ScoreObject):
            val = score_obj.get_cmp_value(metric)
            return val

        return sorted(self.scores, key = sort_key, reverse = True)
    
    def get_best(self, metric: Literal["bleu", "chrf"]):
        return self.sort_scores(metric)[0]

    def print_header(self):
        header_string = f"{'CHECKPOINT':<{COL1_WIDTH}}{'BLEU':<{COL2_WIDTH}}{'CHRF'}"
        print(header_string)
        print(f"{'-'*(len(header_string)+2)}")
    
    def print_best(self, metric: Literal["bleu", "chrf"]):
        cp = self.get_best(metric)
        if metric == "bleu":
            print(f"BEST BLEU SCORE: {cp.folder.name:<{COL1_WIDTH}}{cp.bleu_score:<{COL2_WIDTH}.3f}")
        else:
            print(f"BEST CHRF SCORE: {cp.folder.name:<{COL1_WIDTH}}{cp.chrf_score:<{COL2_WIDTH}.3f}")    
    
    def print_last(self):
        cp = self.scores[-1]
        print(f"{cp.folder.name:<{COL1_WIDTH}}{cp.bleu_score:<{COL2_WIDTH}.2f}{cp.chrf_score:.2f}")


class ScoreManager():
    """Class to manage several ModelManagers, sort and print these"""
    def __init__(self):
        self.models = []

    def sort_models(self, metric: Literal["bleu", "chrf"]):
        def sort_key(model_obj: ModelManager):
            val = model_obj.get_best(metric).bleu_score if metric == "bleu" else model_obj.get_best(metric).chrf_score
            return val

        self.models = sorted(self.models, key = sort_key, reverse = True)
    
    def add_score(self, score: ModelManager):
        self.models.append(score)

    def empty(self):
        self.models.clear()

    def print_header(self, index: bool = True):
        if index:
            header_string = f"{'RANK':<{5}}{'CHECKPOINT':<{COL1_WIDTH}}{'BLEU':<{COL2_WIDTH}}{'CHRF'}"
        else:
            header_string = f"{'CHECKPOINT':<{COL1_WIDTH}}{'BLEU':<{COL2_WIDTH}}{'CHRF'}"
        print(header_string)
        print(f"{'-'*(len(header_string)+2)}")

    def print_best(self, model: ModelManager, rank: int, metric: Literal["bleu", "chrf"] = "bleu"):
        """Print the best checkpoint for the specified model"""
        print(f"{rank:<{5}}{model.model.name}")
        #self.print_header(index=False)
        best_cp = model.get_best(metric)  
        print(f"{best_cp.folder.name:<{COL1_WIDTH}}{best_cp.bleu_score:<{COL2_WIDTH}.3f}{best_cp.chrf_score:.3f}")
        
        print()
        
    def print_score(self, model: ModelManager, rank: int, metric: Literal["bleu", "chrf"] = "bleu", sort: bool = True):
        """Print all checkpoint scores for the specified model"""
        if rank is not None:
            print(f"{rank:<{5}}{model.model.name}")
        else: 
            print(f"{model.model.name}")
        self.print_header()
        checkpoints = model.sort_scores(metric) if sort else model.sort_literal()
        for i, cp in enumerate(checkpoints, 1):
            print(f"{i:<{5}}{cp.folder.name:<{COL1_WIDTH}}{cp.bleu_score:<{COL2_WIDTH}.2f}{cp.chrf_score:.2f}")
        
        print()

    #def print_last(self):
    #    self.print_score(self.scores[-1])

    def print_all(self, metric: Literal["bleu", "chrf"] = "bleu", verbose: bool = False):
        """Print evaluation for all models"""
        self.sort_models(metric)
        for i, model in enumerate(self.models, 1):
            if verbose:
                self.print_score(model, i, metric)
            else:
                self.print_best(model, i, metric)    


def single_folder(args):
    bleu = BLEU()
    chrf = CHRF(word_order=2)
    
    checkpoint_hypotheses = [child for child in Path(args.dir).iterdir() if child.name != "validation.ref" and child.is_file()]
    #sorted_checkpoint_hypotheses = sorted(checkpoint_hypotheses, key = lambda x: int(x.name.split(".")[0][10:]))
    sorted_checkpoint_hypotheses = sorted(checkpoint_hypotheses)
    ref = Path(args.dir) / "validation.ref"

    with open(ref, "r", encoding = "utf-8") as ref_file:
        references = ref_file.read().strip().split("\n")

    model_manager = ModelManager(Path(args.dir))
    model_manager.print_header()
    for hyp in sorted_checkpoint_hypotheses:
        with open(hyp, "r", encoding = "utf-8") as hyp_file:
            hypotheses = hyp_file.read().rstrip().split("\n")

        bleu_score = bleu.corpus_score(hypotheses, [references]).score
        chrf_score = chrf.corpus_score(hypotheses, [references]).score

        score = ScoreObject(hyp, bleu_score, chrf_score)
        model_manager.add_score(score)

        model_manager.print_last()

    print()
    model_manager.print_best("bleu")
    model_manager.print_best("chrf")

    return model_manager

def multi_folder(args):
    bleu = BLEU()
    chrf = CHRF(word_order=2)

    score_manager = ScoreManager()

    #folders = [f for f in Path(args.dir).iterdir() if f.name.startswith("nb_nn_multi_simple") or f.name.startswith("nb_nn_ms")]
    folders = [f for f in Path(args.dir).iterdir() if f.is_dir()]
    
    for folder in tqdm(folders):
        checkpoint_hypotheses = [child for child in folder.iterdir() if child.name != "validation.ref" and child.is_file()]
        #sorted_checkpoint_hypotheses = sorted(checkpoint_hypotheses, key = lambda x: int(x.name.split(".")[0][10:]))
        ref = folder / "validation.ref"

        with open(ref, "r", encoding = "utf-8") as ref_file:
            references = ref_file.read().strip().split("\n")
    
        model_manager = ModelManager(folder)
        for hyp in checkpoint_hypotheses:
            with open(hyp, "r", encoding = "utf-8") as hyp_file:
                hypotheses = hyp_file.read().rstrip().split("\n")
    
            bleu_score = bleu.corpus_score(hypotheses, [references]).score
            chrf_score = chrf.corpus_score(hypotheses, [references]).score
    
            score = ScoreObject(hyp, bleu_score, chrf_score)
            model_manager.add_score(score)

        score_manager.add_score(model_manager)

    if args.pickle:
        with open(Path(args.dir) / "score_manager.pickle", "wb") as f:
            pickle.dump(score_manager, f)
    
    score_manager.print_all(verbose = args.verbose)

    return score_manager

if __name__ == "__main__":
    args = parse_args()
    if args.multi:
        score_manager = multi_folder(args)
    else:
        model_manager = single_folder(args)

    
