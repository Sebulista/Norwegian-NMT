""" Script for processing the NCC dataset to extract Nynorsk data """
import json
import random

from pathlib import Path
from collections import Counter
from typing import Literal
from tqdm import tqdm

from sentence_splitter import SentenceSplitter

def load_jsonl(filename):
    with open(filename, 'r', encoding = "utf-8") as file:
        return [json.loads(line) for line in file]

def compute_distribution(data):
    doc_types = [entry['doc_type'] for entry in data if "maalfrid" not in entry['doc_type']]
    return Counter(doc_types)

def language_filter(data, language: Literal["no", "nn"], confidence_threshold: float = 0.75):
    # Filter NO on 0.9 ~17 000 per shard
    # Filter NN on 0.75 ~2 500 per shard
    filtered_data = [entry for entry in data if entry["lang_fasttext"] == language and float(entry["lang_fasttext_conf"]) >= confidence_threshold]

    return filtered_data

def stratified_sample(data, distribution, sample_fraction=0.1):
    sampled_data = []
    for doc_type, count in distribution.items():
        # Calculate how many samples to take for this doc_type
        num_samples = int(sample_fraction * count)
        
        # Filter data for this doc_type and sample randomly from it
        doc_type_data = [entry for entry in data if entry['doc_type'] == doc_type]
        sampled_data.extend(random.sample(doc_type_data, min(num_samples, len(doc_type_data))))
    
    return sampled_data


def _segment(lang: Literal["en", "no"], sents: str):
    """
    Split sentences using the heuristic sentence splitter

    Args:
        lang (str): language tag: "en", "no"
        sents (str): text string containing sentences to split 
    """
    segmenter = SentenceSplitter(language = lang)

    sentence_segments = json.loads(json.dumps(segmenter.split(sents)))

    return sentence_segments

def split_sentences(entries: dict, lang: Literal["en", "nn", "no"]):
    """
    Split sentences in a text string (document)
    """
    language = "no" if lang == "nn" else lang

    new_sents = []
    for entry in entries:
        sents = _segment(language, entry["text"])
        # hvorfor? er det \n i setningene?
        for sent in sents:
            split = sent.split("\n")
            for s in split:
                new_sents.append(s)

    return new_sents



def process_shard(shard: Path, threshold: float = 0.75):
    data = load_jsonl(shard)
    #doc_distribution = compute_distribution(data)
    #sample = stratified_sample(data, doc_distribution, 0.1)
    sample = data
    lang_filter = language_filter(sample, "nn", threshold)
    sentences = split_sentences(lang_filter, "nn")

    return sentences


if __name__ == "__main__":
    source_dir = Path("path/to/NCC")
    jsonl_dir = source_dir / "jsonl_data"
    processed_dir = source_dir / "raw_data"

    all_sents = 0
    for child in tqdm(list(jsonl_dir.iterdir())):
        if child.is_file():
            sents = process_shard(child, 0.0)
            all_sents += len(sents)
    print(f"Number of sentences: {all_sents}")
            
