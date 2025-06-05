""" CLI program to run bifixer on a folder of bitext files """
import subprocess
import shlex
from pathlib import Path
from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--source_dir", "-s", type = str, help = "Folder in which to bifix the files")
    parser.add_argument("--outdir", "-o", type = str, help = "Output folder")

    args = parser.parse_args()

    return args

def bifix(path, args):
    output_path = Path(args.outdir)
    (output_path / path.name).mkdir(exist_ok = True)
    for file in path.iterdir():
        input = path / file.name
        output = output_path / args.source_dir / file.name
        slang = args.source_dir.split("_")[1]
        tlang = args.source_dir.split("_")[0]

        if not output.exists():
            subprocess.call(shlex.split(f"bifixer {input} {output} {slang} {tlang} --scol 1 --tcol 2 --ignore_duplicates --annotated_output --words_before_segmenting 30 --segmenter heuristic"))
        else:
            print(f"Skipping {output} it has been processed already.")

def main():
    args = parse_arguments()
    bifix(Path(args.source_dir), args)

if __name__== "__main__":
    main()
    
