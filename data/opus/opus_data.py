""" Script for querying the OPUS API for information about available sentence pairs in the
various formats provided for each dataset within the specified language direction """
import requests
import json
import pandas
import os

BOKMÅL = ["no", "nb", "no_nb", "nb_no", "nb_NO"]
NYNORSK = ["nn", "nn_no", "nn_NO"]
#ENGELSK  = ["en", "en_AU", "en_GB", "en_US", "en_CA", "en_NZ", "en_ZA", "en_ca", "en_gb", "en_za"]
ENGELSK  = ["en_AU", "en_GB", "en_US", "en_CA", "en_NZ", "en_ZA", "en_ca", "en_gb", "en_za"]
FORMATS = ["xml", "moses", "tmx"]

dataframe = pandas.DataFrame({
    "corpus":[],
    "xml":[],
    "moses":[],
    "tmx":[]
})

source = "nn"
target = "en_CA"
#format = "moses"

url_start = "https://opus.nlpl.eu/opusapi/?"
url = url_start + f"source={source}" + f"&target={target}" + f"&preprocessing={format}" + f"&version=latest"

response = requests.get(url)
html = response.text
data = json.loads(html)

"""for format in FORMATS:
    for corpus in data["corpora"]:
        row = {"corpus":"", "xml":0, "moses":0, "tmx":0}
        if (corpus["source"] == source or corpus["source"] == target) and (corpus["target"] == target or corpus["target"] == source):
            pass
        #df.loc[len(df.index)]:"""


def get_corpora(source, target):
    url_start = "https://opus.nlpl.eu/opusapi/?"
    url = url_start + f"corpora=True" + f"&source={source}" + f"&target={target}"
    response = requests.get(url)
    html = response.text
    corpora = json.loads(html)["corpora"]

    return corpora

def get_data (source, target, corpus, formats):
    data = []
    url_start = "https://opus.nlpl.eu/opusapi/?"
    #url = url_start + f"corpus={corpus}" + f"&source={source}" + f"&target={target}" + f"&preprocessing={format}" + f"&version=latest"
    url = url_start + f"corpus={corpus}" + f"&source={source}" + f"&target={target}" + f"&version=latest"
    print(f"opening: {url}")
    response = requests.get(url)
    html = response.text
    raw_data = json.loads(html)["corpora"]
    for row in raw_data:
        if (row["source"] == source or row["source"] == target) and (row["target"] == target or row["target"] == source) and row["preprocessing"] in formats:
            data.append(row)

    return data

def new_name(source, target, rows_list, data):
    name = data[0]["corpus"]
    for row in rows_list:
        #print(row)
        if name == row[0]:
            name = f"{name}-{source}-{target}"
            return name, False
    return name, True

def make_frame(sources, targets, formats, filename):
    rows_list = []
    rows_list_unique = []
    for source in sources:
        for target in targets:
            for corpus in get_corpora(source, target):
                data = get_data(source, target, corpus, formats)
                if len(data) == 0:
                    continue
                name, unique = new_name(source, target, rows_list, data)
                temp = [name]
                for row in data:
                    temp.append(row["alignment_pairs"])
                if unique:
                    rows_list_unique.append(temp)
                rows_list.append(temp)

    dir = "opus/"

    if not os.path.exists(dir):
        os.mkdir(dir)


    df = pandas.DataFrame(rows_list, columns = ["corpus", "xml", "moses", "tmx"])
    #df.sort_values(by="corpus", inplace=True)
    #df.loc["Total"] = pandas.Series(df[["xml", "moses", "tmx"]].sum(), index=["xml", "moses", "tmx"])


    #df.to_csv(os.path.join(dir, f"{filename}.csv"), sep=",", index=False, encoding="utf-8")
    print(df)

    df_unique = pandas.DataFrame(rows_list_unique, columns = ["corpus", "xml", "moses", "tmx"])
    #df_unique.sort_values(by="corpus", inplace=True)
    #df_unique.loc["Total"] = pandas.Series(df_unique[["xml", "moses", "tmx"]].sum(), index=["xml", "moses", "tmx"])
    
    #df_unique.to_csv(os.path.join(dir, f"{filename}_unique.csv"), sep=",", index=False, encoding="utf-8")
    print(df_unique)

#make_frame(BOKMÅL, NYNORSK, FORMATS, "bm-nn")
make_frame(BOKMÅL, ENGELSK, FORMATS, "bm-en")
make_frame(NYNORSK, ENGELSK, FORMATS, "nn-en")
