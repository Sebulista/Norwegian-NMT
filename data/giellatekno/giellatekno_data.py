"""
This program searches through the giellatekno files and creates document alignments by URLs
"""

import requests
#import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup, NavigableString
import os
import nltk

# Make sure the required packages are downloaded
nltk.download('punkt')

def get_response(url):
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    return soup



links = []
sentence_counts = {"nno":0, "nob":0, "eng":0}

url_start = "https://gtsvn.uit.no/freecorpus/converted/"

language_codes = ["nno", "nob", "eng"]

url_nno = url_start + "nno/"
url_nob = url_start + "nob/"
url_eng = url_start + "eng/"


#create url to parallel data
def url_builder(url, lang, end):
    url_split = url.split("/")
    url_split[5] = lang
    url_split[-1] = end

    return ("/".join(url_split))+".xml"

#get name of corpus
def get_corpus_name(url):
    url_split = url.split("/")
    corpus = f"{url_split[-3]}/{url_split[-2]}"

    return corpus

#write source and target urls to file
def write_to_file(source, targets):
    corpus = get_corpus_name(source)

    dir = f"../giellatekno/{corpus}"

    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    path = os.path.join(dir, "s-t.txt")

    with open(path, "a") as f:
        f.write(f"SOURCE: {source}\n")
        for target in targets:
            f.write(f"TARGET: {target}\n")
        f.write("\n")

#recursively find xml documents
def find_links(url):
    if url.endswith(".xml"):
        targets = find_parallel_text(url)
        if len(targets):
            """print(f"SOURCE: {url}")
            for target in targets:
                print(f"TARGET: [{target}]")
            print()"""
            #print(f"processing:{url}")
            #write_to_file(url, targets)
    else:
        continuations = get_response(url).find_all("a", href=True)[1:]
        for continuation in continuations:
            cont = continuation["href"]
            find_links(url+cont)

#count sentence pairs
def count_sentence_pairs_rec(url, lang):
    if url.endswith(".xml"):
        targets = find_parallel_text(url)
        if len(targets):
            print(f"processing:{url}")
            text = get_xml_text(url)
            sentences = split_sentences(text, lang)
            sentence_counts[lang] += len(sentences)
    else:
        continuations = get_response(url).find_all("a", href=True)[1:]
        for continuation in continuations:
            cont = continuation["href"]
            count_sentence_pairs_rec(url+cont, lang)

def count_sentence_pairs(urls, lang):
    sents = 0

    for url in urls:
        print(f"processing:{url}")
        text = get_xml_text(url)
        sentences = split_sentences(text, lang)
        sents += len(sentences)

    return sents

#get relevent xml body text
def get_xml_text(url):
    response = requests.get(url)
    response.encoding = 'utf-8'
    xml_content = response.text

    soup = BeautifulSoup(xml_content, "lxml-xml")

    body = soup.find("body")
    for element in body.find_all(attrs={"xml:lang": True}):
        try:
            if element["xml:lang"] not in language_codes+["dan"]:
                element.decompose()
        except TypeError:
            print("\n--------------TYPEERROR-------------\n")

    stripped_text = "\n".join([sent for sent in body.get_text().split("\n") if sent])

    return stripped_text

#split sentences
def split_sentences(text, lang="eng"):
    language = "english"

    if lang == "nno":
        language = "norwegian"
    elif lang == "nob":
        language = "norwegian"

    if language == "norwegian":
        no_abbreviations = ["ca", "dvs", "f.eks", "o.l", "osv", "m.m", "tlf", "postnr", "nr", "pst", "jf", "evt", "inkl", "eks", "vha", "vedr", "mht", "p.t", "a.s", "mv", "arbeidsgodtgj", "bl.a", "mva", "el"]
        tokenizer = nltk.data.load('tokenizers/punkt/norwegian.pickle')
        tokenizer._params.abbrev_types.update(no_abbreviations)

    else:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    tokenizer = nltk.data.load('tokenizers/punkt/{0}.pickle'.format(language))
    sents = tokenizer.tokenize(text)
    new_sents = []
    for sent in sents:
        split = sent.split("\n")
        for s in split:
            new_sents.append(s)

    return new_sents

#find relevant parallel text urls
def find_parallel_text(url):
    response = requests.get(url)
    response.encoding = 'utf-8'
    xml_content = response.text

    soup = BeautifulSoup(xml_content, "lxml-xml")
    parallel = soup.find_all("parallel_text")
    parallel_urls = []
    for p in parallel:
        if p["xml:lang"] in language_codes:
            parallel_url = url_builder(url, p["xml:lang"], p["location"])
            parallel_urls.append(parallel_url)

    return parallel_urls



#count_sentence_pairs_rec("https://gtsvn.uit.no/freecorpus/converted/nno/admin/gonagasviessu.no/", "nno")

count_sentence_pairs_rec(url_nno, "nno")
count_sentence_pairs_rec(url_nob, "nob")
count_sentence_pairs_rec(url_eng, "eng")

print(sentence_counts)
"""
print(f"Nynorsk count: {nno_count}")
print(f"Bokmål count: {nob_count}")
print(f"English count: {eng_count}")"""
#print(count_sentence_pairs("https://gtsvn.uit.no/freecorpus/converted/nno/admin/gonagasviessu.no/", "nno"))



#find_links(url_nno)
"""process_xml("https://gtsvn.uit.no/freecorpus/converted/nno/admin/depts/regjeringen.no/nytt-hovudstyre-i-noregs-bank_id_664718.html.xml")
print()
process_xml("https://gtsvn.uit.no/freecorpus/converted/nno/admin/gonagasviessu.no/nyhet.html_tid=112917.html.xml")
print()
process_xml("https://gtsvn.uit.no/freecorpus/converted/eng/admin/gonagasviessu.no/nyhet.html_tid=112917.html.xml")"""
#print(get_xml_text("https://gtsvn.uit.no/freecorpus/converted/eng/admin/gonagasviessu.no/nyhet.html_tid=112917.html.xml"))
