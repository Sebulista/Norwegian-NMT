""" Utility program to inspect preference dataset in a legible manner """
import json

import regex as re
from pathlib import Path
from argparse import ArgumentParser
from typing import Literal

class SentencePair():    
    def __init__(self, source: str, chosen: str, reject: str, max_score: float, min_score: float, tol: float):
        self.source = source
        self.chosen = chosen
        self.reject = reject
        #self.max_score = max_score
        #self.min_score = min_score
        self.delta = max_score - min_score

        # Shouldn't be here
        self.tol = tol
    
    def meets_requirements(self):
        return (not len({chosen, reject}) == 1) and self.delta > self.tol
    
    def print_line(self):
        print(f"<SOURCE> {self.source}\n")
        print(f"<CHOSEN> {self.chosen[:-1]}")
        print(f"<REJECT> {self.reject[:-1]}")
        print(f"<DELTA> {self.delta:.5f}\n{'-'*100}")

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("file", help = "Which file to process")
    parser.add_argument("--sort", "-s", action = "store_true", help = "Whether or not to sort the output")
    parser.add_argument("--reversed", "-r", action = "store_true", help = "Whether or not to print sentence pairs in reversed order")
    parser.add_argument("--quiet", "-q", action = "store_true", help = "Whether or not to not print sentence pairs")
    parser.add_argument("--tol", "-t", type = float, default = 1e-8, help = "Similarity tolerance.")

    return parser.parse_args()

def get_source_target(file: Path):
    pat = re.compile(r"(en|nn|nb)_(en|nn|nb)")
    match = re.search(pat, file.name)
    src, tgt = None, None
    
    if match is not None:
        src, tgt = match.group().split("_")

    return src, tgt

def get_chosen_reject(example: dict[str, str], target_lang: str, cpo_scorer: Literal["kiwi", "comet"] = "kiwi") -> tuple[str, str]:
    """Given an example entry and target lang, return the chosen and rejected sentences"""
    high_score_key = f"high_{target_lang}_{cpo_scorer}"
    low_score_key = f"low_{target_lang}_{cpo_scorer}"
    ref_score_key = f"ref_{target_lang}_{cpo_scorer}"

    high_output_key = f"high_{target_lang}"
    low_output_key = f"low_{target_lang}"
    ref_output_key = target_lang

    # Defining the sentences and their scores
    sentences = [example[ref_output_key], example[high_output_key], example[low_output_key]]
    scores = [example[ref_score_key], example[high_score_key], example[low_score_key]]

    # ref_score = -1.0 if not in use, remove it if that's the case
    if scores[0] < 0:
        sentences = sentences[1:]
        scores = scores[1:]

    # Finding the indexes for the highest and lowest scores
    highest_score_index = scores.index(max(scores))
    lowest_score_index = scores.index(min(scores))

    # Assigning the corresponding sentences
    highest_score_sentence = f"{sentences[highest_score_index]}\n"
    lowest_score_sentence = f"{sentences[lowest_score_index]}\n"
    
    return highest_score_sentence, lowest_score_sentence, max(scores), min(scores)
        

if __name__ == "__main__":
    args = parse_args()

    file = Path(args.file)
    src, tgt = get_source_target(file)

    assert file.is_file()
    assert file.suffix == ".jsonl"
    assert src is not None
    
    with open(args.file, "r", encoding = "utf-8") as in_file:
        lines = in_file.read().strip().split("\n")
    
    data = [json.loads(line) for line in lines]
    pairs = []
    counter = 0
    for d in data:
        source = d["translation"][src]
        chosen, reject, max_score, min_score = get_chosen_reject(d["translation"], tgt)

        pair = SentencePair(source, chosen, reject, max_score, min_score, args.tol)

        if not pair.meets_requirements():
            continue

        if not args.quiet:
            if not args.sort:
                pair.print_line()
            else:
                pairs.append(pair)
        
        """if len({chosen, reject}) == 1:
            continue

        print(f"<SOURCE> {source}\n")
        print(f"<CHOSEN> {chosen[:-1]}")
        print(f"<REJECT> {reject[:-1]}")
        print(f"<DELTA> {(max_score - min_score):.5f}\n{'-'*100}")"""

        counter += 1

    if not args.quiet:
        if args.sort:
            sorted_pairs = sorted(pairs, key = lambda x: x.delta, reverse = args.reversed)
            for line in sorted_pairs:
                line.print_line()
        
    print(f"Printed {counter} lines")
