""" Script for converting nasjonalbiblioteket data into moses format (plain text source\ttarget) """
import xml.etree.ElementTree as ET
import tarfile
import re
import gzip

def traverse_amesto():
    with tarfile.open("../../data/nb/2018_tm_amesto.tar.gz", "r:gz") as archive:
        src = ""
        trg = ""
        for member in archive.getmembers():
            if member.isdir():
                src, trg = member.name.split("_")
            elif member.isfile():
                file = archive.extractfile(member)
                #print(f"File: {member.name}")
                #content = file.read()
                #src_filename = "2018_tm_amesto_"
                path = "../../data/nb/moses/2018_tm_amesto/"
                src_filename = f"{path}2018_tm_amesto_{member.name.split('/')[1].split('_')[0]}.{src}-{trg}.{src}"
                trg_filename = f"{path}2018_tm_amesto_{member.name.split('/')[1].split('_')[0]}.{src}-{trg}.{trg}"

                print(f"Processing: {member.name}")

                tmx_to_moses(file, src, trg, src_filename, trg_filename)

def traverse_standard():
    with tarfile.open("../../data/nb/2018_tm_standard_norge.tar.gz") as archive:
        src = "en"
        trg = "nb"
        for member in archive.getmembers():
            if member.isfile():
                file = archive.extractfile(member)
                path = "../../data/nb/moses/"
                src_filename = f"{path}{'2018-standard-norge'}.{src}-{trg}.{src}"
                trg_filename = f"{path}{'2018-standard-norge'}.{src}-{trg}.{trg}"

                print(f"Processing: {member.name}")

                tmx_to_moses(file, src, trg, src_filename, trg_filename)

def traverse_doffin():
    with tarfile.open("../../data//train-parts/nasjonalbiblioteket/2020_doffin_tm.tar.gz") as archive:
        src = ""
        trg = "en"
        mapping = {"nob": "nb", "nno" : "nn"}
        for member in archive.getmembers():
            if (member.isfile() and member.name.split(".")[-1] == "tmx"):
                src = mapping[member.name.split(".")[0].split("_")[-1]]
                file = archive.extractfile(member)
                path = "../../data/nb/moses/"
                src_filename = f"{path}{'2020_doffin_tm'}.{src}-{trg}.{src}"
                trg_filename = f"{path}{'2020_doffin_tm'}.{src}-{trg}.{trg}"

                print(f"Processing: {member.name}")

                tmx_to_moses(file, src, trg, src_filename, trg_filename)

def traverse_semantix():
    with tarfile.open("../../data/nb/2020_semantix_tm.tar.gz") as archive:
        src = ""
        trg = ""
        mapping = {"nob": "nb", "nno" : "nn", "eng" : "en"}
        for member in archive.getmembers():
            if (member.isfile() and member.name.split(".")[-1] == "xml"):
                src, trg = member.name.split("-")[1:3]
                src = mapping[src]
                trg = mapping[trg]
                file = archive.extractfile(member)
                path = "../../data/nb/moses/"
                src_filename = f"{path}{'2020_semantix_tm'}.{src}-{trg}.{src}"
                trg_filename = f"{path}{'2020_semantix_tm'}.{src}-{trg}.{trg}"

                print(f"Processing: {member.name}")

                tmx_to_moses(file, src, trg, src_filename, trg_filename)

def traverse_sjodir():
    with tarfile.open("../../data/nb/2020_sjodir_tm.tar.gz") as archive:
        src = "nb"
        trg = "en"
        for member in archive.getmembers():
            if (member.isfile() and member.name.split(".")[-1] == "tmx"):
                file = archive.extractfile(member)
                path = "../../data/nb/moses/"
                src_filename = f"{path}{'2020_sjodir_tm'}.{src}-{trg}.{src}"
                trg_filename = f"{path}{'2020_sjodir_tm'}.{src}-{trg}.{trg}"

                print(f"Processing: {member.name}")

                tmx_to_moses(file, src, trg, src_filename, trg_filename)

def traverse_ud():
    with tarfile.open("../../data/nb/2020_ud_tm.tar.gz") as archive:
        src = ""
        trg = "en"
        mapping = {"nob": "nb", "nno" : "nn"}
        for member in archive.getmembers():
            if (member.isfile() and member.name.split(".")[-1] == "tmx"):
                src = mapping[member.name.split(".")[0].split("_")[-1]]
                file = archive.extractfile(member)
                path = "../../data/nb/moses/"
                src_filename = f"{path}{'2020_ud_tm'}.{src}-{trg}.{src}"
                trg_filename = f"{path}{'2020_ud_tm'}.{src}-{trg}.{trg}"

                print(f"Processing: {member.name}")

                tmx_to_moses(file, src, trg, src_filename, trg_filename)

def traverse_efta():    
    with tarfile.open("../../data/nb/2021_efta_tm.tar.gz") as archive:
        src = ""
        trg = ""
        mapping = {"nob": "nb", "nno" : "nn", "eng" : "en"}
        for member in archive.getmembers():
            if (member.isfile() and member.name.split(".")[-1] == "tmx"):
                src, trg = member.name.split(".")[1].split("_")[2:4]
                src = mapping[src]
                trg = mapping[trg]
                file = archive.extractfile(member)
                path = "../../data/nb/moses/"
                src_filename = f"{path}{'2021_efta_tm'}.{src}-{trg}.{src}"
                trg_filename = f"{path}{'2021_efta_tm'}.{src}-{trg}.{trg}"

                print(f"Processing: {member.name}")

                tmx_to_moses(file, src, trg, src_filename, trg_filename)

def traverse_offweb():
    with tarfile.open("../../data/nb/20180402_offweb.tar.gz", "r:gz") as archive:
        src = ""
        trg = ""
        mapping = {"nob": "nb", "nno" : "nn", "eng" : "en"}
        for member in archive.getmembers():
            if member.isdir() and len(member.name.split("-"))>1:
                src, trg = member.name.split("-")[1:3]
                src = mapping[src]
                trg = mapping[trg]
            elif (member.isfile() and member.name.split(".")[-1] == "xml"):
                file = archive.extractfile(member)
                path = "../../data/nb/moses/20180402_offweb/"
                src_filename = f"{path}{member.name.split('/')[-1].split('-')[0]}.{src}-{trg}.{src}"
                trg_filename = f"{path}{member.name.split('/')[-1].split('-')[0]}.{src}-{trg}.{trg}"
                print(f"Pocessing: {member.name}")

                tree = ET.parse(file)
                root = tree.getroot()

                
                with open(src_filename, "w", encoding="utf-8") as src_file, \
                        open(trg_filename, "w", encoding="utf-8") as trg_file:

                    for tu in root.findall("document/senalign"):
                        sent = tu.findall("lang/s")
                        src_sent = sent[0].text
                        trg_sent = sent[1].text

                        if src_sent == None or trg_sent == None or src_sent.strip() == "" or trg_sent.strip() == "":
                            continue

                        try:
                            clean_src = src_sent.replace("\n", "").replace("\r", "") + "\n"
                            clean_trg = trg_sent.replace("\n", "").replace("\r", "") + "\n"

                            src_file.write(clean_src)
                            trg_file.write(clean_trg)
                        except TypeError as error:
                            print("TypeError")

                assert get_line_count(src_filename) == get_line_count(trg_filename)

def traverse_maalfrid():
    with tarfile.open("../../data/nb/20221223_maalfrid_tmx.tar.gz", "r:gz") as archive:
        src = ""
        trg = ""
        for member in archive.getmembers():
            if member.isfile():
                file = archive.extractfile(member)
                with gzip.open(file, "rb") as gz_file:
                    raw_data = gz_file.read()
                    text_data = raw_data.decode("utf-8")

                path = "../../data/nb/moses/20221223_maalfrid/"

                src = member.name.split('/')[1].split('-')[-2]
                trg = member.name.split('/')[1].split('-')[-1].split('.')[0]

                src_filename = f"{path}{'.'.join(member.name.split('/')[1].split('.')[0:2])}.{src}-{trg}.{src}"
                trg_filename = f"{path}{'.'.join(member.name.split('/')[1].split('.')[0:2])}.{src}-{trg}.{trg}"

                print(f"Processing: {member.name}")

                tmx_to_moses(text_data, src, trg, src_filename, trg_filename, from_string = True)


def tmx_to_moses(tmx_filename, src_language, tgt_language, src_filename, tgt_filename, from_string = False):
    if from_string:
        root = ET.fromstring(tmx_filename)
    else:
        tree = ET.parse(tmx_filename)
        root = tree.getroot()

    # Open files to write source and target languages
    #with open(src_filename, "w", encoding="utf-8") as src_file, \
    #        open(tgt_filename, "w", encoding="utf-8") as tgt_file:

    for tu in root.findall("body/tu"):
        tuv = tu.findall("tuv")
        src_sent = tuv[0].find("seg").text
        trg_sent = tuv[1].find("seg").text

        if src_sent == None or trg_sent == None or src_sent.strip() == "" or trg_sent.strip() == "":
            continue

        try:
            clean_src = src_sent.replace("\n", "").replace("\r", "") + "\n"
            clean_trg = trg_sent.replace("\n", "").replace("\r", "") + "\n"

            #src_file.write(clean_src)
            #tgt_file.write(clean_trg)
        except TypeError as error:
            print("TypeError")
            print(error)
            print(src_sent)
            print(trg_sent)

    #assert get_line_count(src_filename) == get_line_count(tgt_filename)

def get_file_name(tmx_filename):
    split = tmx_filename.split("_")

def get_line_count(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return sum(1 for _ in file)


if __name__ == "__main__":
    traverse_doffin()

