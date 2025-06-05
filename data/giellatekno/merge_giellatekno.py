""" This script merges all the giellatekno subfolder files into one """

from pathlib import Path
from collections import defaultdict

def find_root(path: Path):
    """ Find the folder structure until giellatekno_deduped for a file """
    desired_root = None
    subpath_after_root = None
    for parent in path.parents:
        if parent.name == "giellatekno_deduped":
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

def merge_files(folder: Path, output_folder: Path):
    directions = defaultdict(list)

    for child in folder.iterdir():
        if child.is_file():
            direction = child.suffix
            directions[direction].append(child)

    for direction, files in directions.items():
        with open(output_folder / f"{find_root(folder)}{direction}", "w", encoding = "utf-8") as out_file:
            for file in files:
                with open(file, "r", encoding = "utf-8") as in_file:
                    lines = in_file.read().strip().split("\n")
                for line in lines:
                    if line == "":
                        continue
                    out_file.write(f"{line}\n")
                

def main():
    data_folder = Path("path/to/data")
    source_path = Path(data_folder / "giellatekno_deduped")
    output_path = Path(data_folder / "giellatekno_merged")

    folders = set()

    copy_folders(source_path, output_path, folders)

    for folder in folders:
        merge_files(folder, output_path)

if __name__ == "__main__":
    main()
