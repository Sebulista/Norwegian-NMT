""" CLI program to perform bicleaner hardrules on a folder of bitext files 
and output the annotated decision to a output folder with equal subdirectory structure"""

import subprocess
import shlex
from pathlib import Path
from argparse import ArgumentParser

path = Path("path/to/data")

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--source_dir", "-s", type = str, help = "Folder in which to bifix the files")
    parser.add_argument("--outdir", "-o", type = str, help = "Output folder")

    args = parser.parse_args()

    return args

def apply_hardrules(path, args):
    output_path = Path("path/to/output_dir")
    (output_path / args.source_dir).mkdir(exist_ok = True)
    for file in path.iterdir():
        input = path / file.name
        output = output_path / args.source_dir / file.name
        slang = args.source_dir.split("_")[0]
        tlang = args.source_dir.split("_")[1]

        if not output.exists():
            subprocess.call(shlex.split(f"bicleaner-hardrules {input} {output} --scol 1 --tcol 2 --disable_minimal_length --disable_porn_removal --annotated_output -c config.yaml --disable_lang_ident --metadata path/to/bicleaner-models/{slang}-{tlang}/metadata.yaml"))
            
        else:
            print(f"Skipping {output} it has been processed already.")

def main():
    args = parse_arguments()
    apply_hardrules(path / args.source_dir, args)

if __name__== "__main__":
    main()
    
