""" CLI program to filter hardrules annotated sentence pair into a keep and remove
where the keep files are the sentence pairs that passed the hardrules filter, written
with the sentence pair, and the remove files are the fully annotated lines of the 
sentence pairs that were removed"""

import subprocess
import shlex
from pathlib import Path
from argparse import ArgumentParser

path = Path("path/to/annotated_files")
output_path = Path("path/to/output_dir")

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--direction", "-d", type = str, help = "Language direction for which to filter", choices = ["nb_en", "nn_nb", "nn_en"])

    args = parser.parse_args()

    return args

def filter(path, args):
    #Create SLANG_TLANG subfolders
    #keep
    (output_path / "keep" / args.direction).mkdir(exist_ok = True)
    #remove
    (output_path / "remove" / args.direction).mkdir(exist_ok = True)
    
    for file in path.iterdir():
        input = path / file.name
        keep_output = output_path / "keep" / args.direction / file.name
        remove_output = output_path / "remove" / args.direction / file.name
        #slang = args.source_dir.split("_")[1]
        #tlang = args.source_dir.split("_")[0]

        #in order to prevent overwriting and for easier continuation of interrupted runs
        if not (keep_output.exists() and remove_output.exists()):
            print(f"PROCESSING:  {input.name}")
            
            # Will require 2*N memory where N is the size of the input file, not very memory efficient
            with open(input, "r", encoding = "utf-8") as infile:
                lines = infile.read().split("\n")
            with open(keep_output, "w", encoding = "utf-8") as keep, open(remove_output, "w", encoding = "utf-8") as remove:
                for line in lines:
                    if line == "":
                        continue
                    split = line.split("\t")
                    #form: S_sent | T_sent | bifixer_info | hardrules_score | hardrules_decision
                    if split[4] == "keep":
                        new_line = f"{split[0]}\t{split[1]}\n"
                        keep.write(new_line)
                    #keep lines that will be removed as is by including removal decision (will be further processed)
                    else:
                        remove.write(f"{line}\n")
        else:
            print(f"SKIPPING {keep_output.name} it has been processed already.")

def main():
    args = parse_arguments()
    filter(path / args.direction, args)

if __name__== "__main__":
    main()
    
