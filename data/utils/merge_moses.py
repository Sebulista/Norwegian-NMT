""" Script to perform basic preprocessing and merge gzipped monolingual files """
from pathlib import Path
import gzip
from collections import defaultdict
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", type=str, required=True, help="Language translation direction (f.ex. nb-en)")

    return parser.parse_args()


#Merge two gzipped files into one tab separated file
def merge_gzipped_files(source_file_path, target_file_path, output_file_path):
    """
    Merge two gzipped files into one tab separated file

    Parameters:
        source_file_path (string | path): Path to gzipped source file
        target_file_path (string | path): Path to gzipped target file
        output_file_Path (string | path): Path to where the merged file should be stored
    """

    with gzip.open(source_file_path, "rt", encoding = "utf-8") as source_file, \
        gzip.open(target_file_path, "rt", encoding = "utf-8") as target_file, \
        open(output_file_path, "w", encoding = "utf-8") as output_file:

        for source_line, target_line in zip(source_file, target_file):
            #Preprocess each line
            #remove tabs inside of line
            clean_src = source_line.strip().replace("\t", "")
            clean_trg = target_line.strip().replace("\t", "")
            combined_line = clean_src + "\t" + clean_trg
            output_file.write(combined_line + "\n")


#direction = "nn_en"


if __name__ == "__main__":
    args = parse_arguments()
    direction = args.direction

    path = f"path/to/input_folder/{direction}/"
    p = Path(path)
    out_dir = f"path/to/output_folder"
    out_p = Path(out_dir)

    print(p)
    assert(p.is_dir())
    assert(out_p.is_dir())

    pairs = defaultdict(lambda: [])

    #add pairs of files to be merged to dict
    for child in p.iterdir():
        if child.is_file():
            new_name = ".".join(child.name.split(".")[:-2])
            pairs[new_name].append(child)

    #merge every pair in the dict
    for key, value in pairs.items():
        print(f"Processing: {key}")
        out_path = Path(out_dir + key)
        merge_gzipped_files(value[0], value[1], out_path)
