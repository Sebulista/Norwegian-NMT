""" CLI program to calculate and append the comet kiwi score to a folder of bitext files """
import subprocess
import shlex
from pathlib import Path
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--source_dir", "-s", type = str, help = "Folder in which to bifix the files")
    parser.add_argument("--output_dir", "-o", type = str, help = "Output folder")

    args = parser.parse_args()

    return args


def find_root(path: Path):
    """ Find the folder structure until giellatekno_deduped for a file """
    desired_root = None
    subpath_after_root = None
    for parent in path.parents:
        if parent.name == "giellatekno_final":
            desired_root = parent
            subpath_after_root = path.relative_to(parent)
            break

    return subpath_after_root


def copy_folders(path: Path, output_path: Path, folders: set):
    for child in path.iterdir():
        if child.is_dir():
            subdir = find_root(child)
            if len(subdir.parents) == 1:
                (output_path / subdir).mkdir(exist_ok = True)
                print(f"Making {output_path / subdir}")
            copy_folders(child, output_path, folders)
        
        if child.is_file():
            folders.add(child.parent)


def comet_score(path: Path, args):
    """ Run bifix on each file """
    output_path = Path(args.output_dir)
    #(output_path / args.source_dir).mkdir(exist_ok = True)
    for file in path.iterdir():
        #Recursively apply this to each file
        if file.is_dir():
            subdir = find_root(file)
            (output_path / subdir).mkdir(exist_ok = True)
            comet_score(file, args)
        
        # Run bifixer on the file
        elif file.is_file():
            input = path / file.name
            output = output_path / find_root(file).parent / file.name

            command = f"paste {input} <(cut -f3 <(comet-score -s <(cut -f1 {input}) -t <(cut -f2 {input}) --batch_size 32 --model path/to/comet-models/wmt22-cometkiwi-da/checkpoints/model.ckpt) | cut -d ' ' -f2) > {output}"

            #print(command)
            #print(shlex.split(command))

            subprocess.call(shlex.split(command), shell=True)

def main():
    args = parse_arguments()

    source_path = Path(args.source_dir)

    comet_score(source_path, args)

if __name__== "__main__":
    main()
