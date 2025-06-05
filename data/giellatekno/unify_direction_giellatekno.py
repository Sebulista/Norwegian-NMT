""" This scripts makes sure all bitexts have the same language direction x->y such that
y->x are swapped both in file name and sentence pairs"""

from pathlib import Path
import shutil

DIRECTIONS = [".en-nb", ".en-nn", ".nb-nn"]
to_swap = set(["nb-en", "nn-en", "nn-nb"])


def find_root(path: Path):
    """ Find the folder structure until giellatekno_aligned for a file """
    desired_root = None
    subpath_after_root = None
    for parent in path.parents:
        if parent.name == "giellatekno_bifixed":
            desired_root = parent
            subpath_after_root = path.relative_to(parent)
            break

    return subpath_after_root


def swap_direction(original_path: Path, out_dir: Path):

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


def get_new_name(name: str):
    #filename format xxx.lang1-lang2
    basename = ".".join(name.split(".")[:-1])
    langs = name.split(".")[-1]
    lang1, lang2 = langs.split("-")

    return f"{basename}.{lang2}-{lang1}"


def copy_folders(path: Path, output_path: Path, folders: set):
    for child in path.iterdir():
        if child.is_dir():
            subdir = find_root(child)
            (output_path / subdir).mkdir(exist_ok = True)
            #print(f"Making {output_path / subdir}")
            copy_folders(child, output_path, folders)
        
        if child.is_file():
            folders.add(child.parent)


def process_files(dir: Path, directions: set[str], out_dir: Path):
    for child in dir.iterdir():
        if child.is_file():
            file_direction = child.name.split(".")[-1]
            if file_direction in directions:
                #print(f"Swapping {child}")
                swap_direction(child, out_dir / find_root(child).parent)
            else:
                shutil.copy(child, out_dir / find_root(child).parent)


def main():
    src_folder = Path("path/to/folder")
    source_path = src_folder / "giellatekno_bifixed")
    output_path = src_folder / "giellatekno_bifixed_corrected")

    folders = set()

    copy_folders(source_path, output_path, folders)

    for folder in folders:
        process_files(folder, directions = to_swap, out_dir = output_path)

if __name__ == "__main__":
    main()
