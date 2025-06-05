""" Given aligned document urls, this script downloads the documents stored in xml format
and turns it into legible txt cleaned with a pre-defined set of rules"""

import requests
from bs4 import BeautifulSoup
import os
import nltk
from sentence_splitter import SentenceSplitter
import json
from pathlib import Path
import yaml
from typing import List

import regex as re
pattern = re.compile(r'[A-Za-z]')
regex_numbers = re.compile("[[:digit:]]")
regex_blank = re.compile("[ \u00A0]")

LANGUAGE_CODES =  {"nno": "nn", "nob" : "nb", "eng" : "en"}


def get_names():
    # Hard coded to find valid dcoument names from an earlier run
    # Only read documents with an actual parallel file
    path = Path("path/to/giellatekno_texts_source")

    names = set()

    def rec_get_names(path):
        for child in path.iterdir():
            if child.is_dir():
                rec_get_names(child)
            else:
                names.add(f"{child.parent.parent.name}/{child.parent.name}/{'.'.join(child.name.split('.')[:-1])}")

        return names
    
    return rec_get_names(path)

valid_names = get_names()

def only_numbers(sentence):
    if len(sentence) == 0:
        return True
    
    threshold = 0.35

    return len(regex_numbers.findall(sentence)) / len(sentence) > threshold

def too_short(sentence, threshold = 2):
    return len(regex_blank.findall(sentence)) < threshold


def get_response(url):
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    return soup

def _segment(lang, sents):
    """
    Split sentences using the heuristic sentence splitter

    Args:
        lang (str): language tag: "en", "no"
        sents (str): text string containing sentences to split 
    """
    segmenter = SentenceSplitter(language = lang)

    sentence_segments = json.loads(json.dumps(segmenter.split(sents)))

    return sentence_segments

#split sentences
def split_sentences(text, lang):
    """
    Split sentences in a text string (document)
    """
    language = "en"
    if lang == "nno":
        language = "no"
    elif lang == "nob":
        language = "no"

    sents = _segment(language, text)
    new_sents = []
    for sent in sents:
        split = sent.split("\n")
        for s in split:
            new_sents.append(s)

    return new_sents

#get relevent xml body text
def get_xml_text(url, remove_list = False):
    response = requests.get(url)
    response.encoding = 'utf-8'
    xml_content = response.text

    soup = BeautifulSoup(xml_content, "lxml-xml")

    language = get_doc_lang(url)
    body = soup.find("body")

    #remove lists
    if remove_list:
        for element in body.find_all("list"):
            #if too_short(element.get_text())
            element.decompose()

    #remove italics that are too short or only numbers
    for element in body.find_all(attrs={"type" : "italic"}):
        text = element.get_text()
        if too_short(text) or only_numbers(text):
            element.decompose()

    #remove titles that are too short or only numbers
    for element in body.find_all(attrs={"type" : "title"}):
        text = element.get_text()
        if too_short(text, 3) or only_numbers(text):
            element.decompose()

    #filter based on language tags
    for element in body.find_all(attrs={"xml:lang": True}):
        text = element.get_text()

        try:
            if element["xml:lang"] in ["sme", "sma", "fkv", "smj", "est"]:
                element.decompose()
                continue
            if element["xml:lang"] == "koi":
                #if span, do not remove
                if not element.find("span"):
                    element.decompose()
                    continue

            if language == "eng" and element["xml:lang"] in ["nno", "nob"]:
                element.decompose()
                continue

            if language in ["nno", "nob"] and element["xml:lang"] == "eng":
                element.decompose()
                continue

        except TypeError:
            #print("TYPEERROR")
            pass

        #finally remove it if it is too short or only numbers
        if too_short(text) or only_numbers(text):
           element.decompose()
           continue

    stripped_text = "\n".join([sent for sent in body.get_text().split("\n") if sent])

    return stripped_text


def get_xml_text_config(url, config: dict, exception = False):
    response = requests.get(url)
    response.encoding = 'utf-8'
    xml_content = response.text

    soup = BeautifulSoup(xml_content, "lxml-xml")

    language = get_doc_lang(url)
    body = soup.find("body")

    if exception:
        pass

    #remove lists
    if config["remove_lists"]:
        for element in body.find_all("list"):
            if config["except_list"] and exception:
                lines = [e.get_text() for e in element]
                #remove empty
                lines = "".join([l for l in lines if l]).split("\n")
                #calculate average
                avg = sum([len(l.split()) for l in lines]) / len(lines)
                if avg >= 5.0:
                    continue

            element.decompose()

    #remove italics that are too short or only numbers
    for element in body.find_all(attrs={"type" : "italic"}):
        if config["remove_italics"]:
            element.decompose()
        else:
            text = element.get_text()
            if too_short(text) or only_numbers(text):
                element.decompose()

    #remove titles that are too short or only numbers
    for element in body.find_all(attrs={"type" : "title"}):
        if config["remove_titles"]:
            element.decompose()
        else:
            text = element.get_text()
            if too_short(text, 3) or only_numbers(text):
                element.decompose()

    if config["remove_bold"]:
        for element in body.find_all(attrs={"type": "bold"}):
            element.decompose()


    #filter based on language tags
    for element in body.find_all(attrs={"xml:lang": True}):
        text = element.get_text()

        try:
            if element["xml:lang"] in ["sme", "sma", "fkv", "smj", "est"]:
                element.decompose()
                continue
            if element["xml:lang"] == "koi":
                #if span, do not remove
                if not element.find("span"):
                    element.decompose()
                    continue

            if language == "eng" and element["xml:lang"] in ["nno", "nob"]:
                element.decompose()
                continue

            if language in ["nno", "nob"] and element["xml:lang"] == "eng":
                element.decompose()
                continue

        except TypeError:
            #print("TYPEERROR")
            pass

        if config["filter_language_tags"]:
            #finally remove it if it is too short or only numbers
            if too_short(text) or only_numbers(text):
                element.decompose()
                continue

    if config["remove"][0] != None:
        for element in body:
            text = element.get_text()
            if text in config["remove"]:
                #print(f"REMOVED: {text}")
                element.decompose()


    if config["remove_contains"][0] != None:
        for entry in config["remove_contains"]:
            pattern = re.compile(entry)
            for element in body:
                text = element.get_text()
                if re.search(pattern, text):
                    #print(f"REMOVED: {text}")
                    element.decompose()

    #deduplicate sentences
    sentences = [sent for sent in body.get_text().split("\n") if sent]
    deduplicated_sentences = list(dict.fromkeys(sentences))

    stripped_text = "\n".join(deduplicated_sentences)

    return stripped_text

#get name of corpus
def get_corpus_name(url):
    url_split = url.split("/")
    corpus = f"{url_split[-3]}/{url_split[-2]}"

    return corpus

#get language
def get_doc_lang(url):
    url_split = url.split("//")[-1].split("/")
    lang = url_split[3]

    return lang

def get_lang(name):
    return name.split(".")[-1]

def swap_lang(name, new_lang):
    name = ".".join(name.split(".")[:-1])
    return f"{name}.{new_lang}"

#get document name
def get_doc_name(url):
    name = url.split("/")[-1].split(".")
    if name[-3] in ["eng", "nob", "no"]:
        name = ".".join(name[:-3])
    else:
        name = ".".join(name[:-2])

    return name

processed = []


def remove_duplicates(source: str, targets: List[str]):
    source_split = source.split("\n")
    target_splits = [target.split("\n") for target in targets]

    source_to_remove = set()
    target_to_remove = [set(),set()]

    for i, sent in enumerate(source_split):
        for j in range(len(target_splits)):
            if sent in target_splits[j]:
                idx = target_splits[j].index(sent)
                source_to_remove.add(i)
                target_to_remove[j].add(idx)

    for i in sorted(source_to_remove, reverse=True):
        del source_split[i]
        #print(f"REMOVING: {source_split[i]}")

    for j in range(len(target_splits)):
        for idx in sorted(target_to_remove[j], reverse=True):
            del target_splits[j][idx]

    source_text = "\n".join(source_split)
    target_texts = ["\n".join(t) for t in target_splits]

    return source_text, target_texts

def make_corpora(src_tgt_file, path):
    print(f"Processing {src_tgt_file}")
    with open(src_tgt_file, "r", encoding = "utf-8") as file:
        data = file.read()

    data = data.strip().split("\n\n")

    #load yaml config
    with open ("giellatekno_config.yaml") as f:
        cfg = yaml.safe_load(f)

    for line in data:
        elements = line.split("\n")
        #URL
        source = elements[0].strip().split()[-1]
        #["URLs"]
        targets = [elements[i].strip().split()[-1] for i in range(1, len(elements))]

        if source in processed:
            #print(f"Skipping {source}")
            continue
        else:
            processed.append(source)

        for tgt in targets:
            if tgt in processed:
                #print(f"Skipping {tgt}")
                continue
            else:
                processed.append(tgt)


        doc_name = f"{get_corpus_name(source)}/{get_doc_name(source)}"
        src_name = f"{doc_name}.{LANGUAGE_CODES[get_doc_lang(source)]}"
        target_names = []
        for tgt in targets:
            tgt_name = f"{doc_name}.{LANGUAGE_CODES[get_doc_lang(tgt)]}"
            target_names.append(tgt_name)

        corpus_name = get_corpus_name(source)
        settings = cfg["default"]
        if doc_name not in valid_names:
            print(f"Skipping: {doc_name}")
            continue

        if corpus_name in cfg:
            settings.update(cfg[get_corpus_name(source)])
            #print(settings)
        else:
            print(f"{corpus_name} did not have a config.")

        #change language
        if settings["exceptions"][0] != None and "except_change_lang" in settings:
            if get_doc_name(source) in settings["exceptions"]:
                change_from = settings["except_change_lang"]["from"]
                change_to = settings["except_change_lang"]["to"]
                
                #source
                if get_lang(src_name) == change_from:
                    src_name = swap_lang(src_name, change_to)

                #targets
                for i in range(len(target_names)):
                    if get_lang(target_names[i]) == change_from:
                        target_names[i] = swap_lang(target_names[i], change_to) 

        exception = False
        if settings["exceptions"][0] != None:
            if get_doc_name(source) in settings["exceptions"]:
                exception = True

        #source_text = get_xml_text(source)
        source_text = get_xml_text_config(source, settings, exception = exception)
        target_texts = []
        for tgt in targets:
            #target_text = get_xml_text(tgt)
            target_text = get_xml_text_config(tgt, settings, exception = exception)
            target_texts.append(target_text)

        source_text, target_texts = remove_duplicates(source_text, target_texts)

        #determine acceptable line ratio
        texts_langs = []
        texts_langs.append( (source_text, get_lang(src_name)) )
        for i in range(len(target_names)):
            texts_langs.append( (target_texts[i], get_lang(target_names[i])) )


        #remove bitexts with too big line ratio
        if "depts" in corpus_name and not acceptable_line_ratio(texts_langs, 1.3):
            print(f"Removing {doc_name} due to line ratio")
            continue

        path = Path(path)

        srcpath = path / src_name
        srcpath.parent.mkdir(parents=True, exist_ok=True)
        with srcpath.open("w", encoding = "utf-8") as f:
            for sent in split_sentences(source_text, get_lang(src_name)):
                #remove lines in the config that might have come to be through sentence segmenting
                if settings["remove"][0] != None and sent in settings["remove"]:
                    continue
                f.write(f"{sent}\n")

        for i in range(len(targets)):
            tgtpath = path / target_names[i]
            with tgtpath.open("w", encoding = "utf-8") as f:
                for sent in split_sentences(target_texts[i], get_lang(target_names[i])):
                    if settings["remove"][0] != None and sent in settings["remove"]:
                        continue
                    f.write(f"{sent}\n")

        
def acceptable_line_ratio(texts_langs, threshold = 1.2):
    lenghts = []
    for txt, lng in texts_langs:
        sents = split_sentences(txt, lng)
        lenghts.append(len(sents))

    lenghts.sort(reverse=True)

    #print(lenghts)
    #print(lenghts[0] / lenghts[-1])
    #print()

    for i, l in enumerate(lenghts):
        if l <= 0:
            lenghts[i] = 1

    #if (lenghts[0] / lenghts[-1]) > threshold:
    if (lenghts[0] / lenghts[1]) > threshold:
        return False
    
    return True

def traverse_folders(start_path):
    path = Path(start_path)
    for child in path.iterdir():
        if child.is_dir():
            traverse_folders(child)
        else:
            make_corpora(child)


def main():
    traverse_folders("../../data/test-parts/giellatekno/")

if __name__ == "__main__":
    main()
