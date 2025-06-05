""" Script with functionality to swap either the filename or the filename and the contents
(source and target side) to make sure the filename extension is correct """

import gzip
from pathlib import Path
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--file", type = str, help = "File to swap")
    parser.add_argument("--dir", type = str, help = "Directory in which to process files")
    parser.add_argument("--direction", type = str, help = "The language direction to swap")
    parser.add_argument("--out_dir", type = str, default = None, help = "Directory in which the swapped file is output")

    args = parser.parse_args()

    if args.file and (args.dir or args.direction):
        parser.error("Cannot use '--file' with '--dir' and '--direction'. Use either '--file' or both '--dir' and '--direction'.")

    if not args.file and not (args.dir and args.direction):
        parser.error("If '--file' is not provided, '--dir' and '--direction' must all be specified.")

    return args

def swap_direction(original_path: Path, out_dir = None):
    #original_path = Path(filename)
    if out_dir is None:
        out_dir = original_path.parent
    else:
        out_dir = Path(out_dir)

    with open(original_path, "rt", encoding = "utf-8") as file:
        content = file.readlines()

    new_name = get_new_name(original_path.name)
    new_path = out_dir / new_name

    wrong_lines = 0
    with open(new_path, "wt", encoding = "utf-8") as file:
        for line in content:
            try:
                src, tgt = line.split("\t")
            except ValueError:
                wrong_lines += 1
                #print(repr(line))
            file.write(f"{tgt.strip()}\t{src.strip()}\n")

    print(f"Created: {new_path}, omitted {wrong_lines} wrong lines")

def swap_file_type_direction(path):
    name = path.name
    basename = ".".join(name.split(".")[:-1])
    langs = name.split(".")[-1]
    lang1, lang2 = langs.split("-")

    path.rename(path.with_suffix(f".{lang2}-{lang1}"))

def get_new_name(name):
    #filename format xxx.lang1_lang2
    basename = ".".join(name.split(".")[:-1])
    langs = name.split(".")[-1]
    lang1, lang2 = langs.split("-")

    return f"REVERSED_{basename}.{lang2}-{lang1}"

def process_files(dir: Path, direction: str, out_dir = None):
    for child in dir.iterdir():
        if child.is_file():
            file_direction = child.name.split(".")[-1]
            if file_direction == direction:
                if child.name == f"2020_ud_tm.{direction}":
                    print(f"Swapping {child}")
                    swap_direction(child, Path("path/to/data"))
                else:
                    print(f"Processing {child}")
                    #swap_direction(child, out_dir)
                    swap_file_type_direction(child)

if __name__ == "__main__":
    args = parse_args()
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)

    if args.file is not None:
        swap_direction(Path(args.file), args.out_dir)

    else:
        dir = Path(args.dir)
        
        assert(dir.is_dir())
        if out_dir != None:
            assert(out_dir.is_dir())

        process_files(dir, args.direction, out_dir)
