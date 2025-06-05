""" This script performs bicleaner hardrules on the giellatekno data
Remember to set all paths
bicleaner-hardrules needs to be installed"""

from pathlib import Path
import subprocess
import shlex

src_path = Path("path/to/source_folder")
bicleaner_path = Path("path/to/bicleaner_models")

def find_root(path: Path):
    """ Find the folder structure until giellatekno_merged for a file """
    desired_root = None
    subpath_after_root = None
    for parent in path.parents:
        if parent.name == "giellatekno_merged":
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

def clean(folder: Path, output_folder: Path):
    for child in folder.iterdir():
        if child.is_file():
            with open(output_folder / f"{find_root(folder)}/{child.name}", "w", encoding = "utf-8") as out_file, \
                open(output_folder / f"annotated_{child.stem}{child.suffix}", "w", encoding = "utf-8") as annotated_out_file:

                print(f"Processing {child}")
                direction = child.name.split(".")[-1]
                
                src, tgt = direction.split("-")
                metadata = bicleaner_path / f"{direction}/metadata.yaml"
                proc = subprocess.run(shlex.split(f"bicleaner-hardrules {child} - --scol 1 --tcol 2 --annotated_output -s {src} -t {tgt} --metadata {metadata} --disable_lang_ident --disable_porn_removal"), stdout=subprocess.PIPE, text = True)

                lines = proc.stdout.strip()
                if lines == "":
                    continue
                lines = lines.split("\n")

                for line in lines:
                    if line.strip() == "":
                        continue

                    split = line.split("\t")
                    if split[2] == "1":
                        out_file.write(f"{split[0]}\t{split[1]}\n")
                    else:
                        annotated_out_file.write(f"{line}\n")


def main():
    source_path = src_path / "giellatekno_merged")
    output_path = src_path / "giellatekno_cleaned")

    folders = set()

    copy_folders(source_path, output_path, folders)

    for folder in folders:
        clean(folder, output_path)

if __name__ == "__main__":
    main()
