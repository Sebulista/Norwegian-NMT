"""This script aligns the parallel corpora at the sentence level using vecalign
Assumes vecalign is installed: https://github.com/thompsonb/vecalign"""
from pathlib import Path

from tempfile import TemporaryDirectory
import subprocess
import shlex


source = "path/to/texts"

CODES = ["en", "nb", "nn"]

src_path = Path(source)

processed = []

folders = []
def traverse(source):
    for child in source.iterdir():
        if child.is_dir():
            traverse(child)
        else:
            folders.append(child.parent)
            break

def get_name(path):
    name = path.name.split(".")[0]
    return name

def find_targets(path):
    lang = path.name.split(".")[-1]
    name = path.name
    corpus = get_name(path)

    targets = []

    for lang_code in CODES:
        if lang_code != lang:
            target = f"{corpus}.{lang_code}"
            target_path = path.parent / target
            if target_path.exists():
                targets.append(target_path)

    return targets


def find_root(path):
    desired_root = None
    subpath_after_root = None
    for parent in path.parents:
        if parent.name == "giellatekno_texts":
            desired_root = parent
            subpath_after_root = path.relative_to(parent)
            break

    return subpath_after_root.parent

def process_folder(folder_source_path):
    processed = []

    with TemporaryDirectory() as temp_dir:

        for child in folder_source_path.iterdir():
            assert(child.is_file())
            name = get_name(child)

            if name in processed:
                #print(f"Skipping {child.name} as it has already been processed")
                continue

            processed.append(name)

            #copy subdirectory structure to temp
            #temp_root = Path("/mnt/d/Master/data/test-parts/giellatekno_temp/")
            subdirs = find_root(child)
            subdirs = temp_dir / subdirs
            subdirs.mkdir(parents=True, exist_ok=True)

            #copy subdirectory structure to align
            align_root = Path("path/to/giellatekno_aligned/")
            align_subdirs = find_root(child)
            align_subdirs = align_root / align_subdirs
            align_subdirs.mkdir(parents=True, exist_ok=True)

            targets = find_targets(child)


            #make overlap and embed them files
            input = child
            out_file = f"{child.name}.overlap"
            output = subdirs / out_file

            #make overlaps
            subprocess.call(shlex.split(f"./overlap.sh {input} {output}"))
            #embed
            input_emb = output
            emb_out_file = f"{output.name}.emb"
            output_emb = subdirs / emb_out_file
            subprocess.call(shlex.split(f"./embed.sh {input_emb} {output_emb}"))

            to_align = []
            to_align.append((child, output, output_emb, child.name.split(".")[-1]))

            for target in targets:
                input = target
                out_file = f"{target.name}.overlap"
                output = subdirs / out_file

                #make overlaps
                subprocess.call(shlex.split(f"./overlap.sh {input} {output}"))
                #embed
                input_emb = output
                emb_out_file = f"{output.name}.emb"
                output_emb = subdirs / emb_out_file
                subprocess.call(shlex.split(f"./embed.sh {input_emb} {output_emb}"))

                to_align.append((target, output, output_emb, target.name.split(".")[-1]))

            #align
            if len(to_align) == 2:
                output_align_file = f"{get_name(to_align[0][0])}.{to_align[0][3]}-{to_align[1][3]}"
                output_align = align_subdirs / output_align_file
                subprocess.call(shlex.split(f"./vecalign.sh {to_align[0][0]} {to_align[1][0]} {to_align[0][1]} {to_align[0][2]} {to_align[1][1]} {to_align[1][2]} {output_align}"))
            
            elif len(to_align) == 3:
                s = 0
                t = 1
                #0 -> 1
                output_align_file = f"{get_name(to_align[s][0])}.{to_align[s][3]}-{to_align[t][3]}"
                output_align = align_subdirs / output_align_file
                subprocess.call(shlex.split(f"./vecalign.sh {to_align[s][0]} {to_align[t][0]} {to_align[s][1]} {to_align[s][2]} {to_align[t][1]} {to_align[t][2]} {output_align}"))
                #0 -> 2
                s = 0
                t = 2
                output_align_file = f"{get_name(to_align[s][0])}.{to_align[s][3]}-{to_align[t][3]}"
                output_align = align_subdirs / output_align_file
                subprocess.call(shlex.split(f"./vecalign.sh {to_align[s][0]} {to_align[t][0]} {to_align[s][1]} {to_align[s][2]} {to_align[t][1]} {to_align[t][2]} {output_align}"))
                #1 -> 2
                s = 1
                t = 2
                output_align_file = f"{get_name(to_align[s][0])}.{to_align[s][3]}-{to_align[t][3]}"
                output_align = align_subdirs / output_align_file
                subprocess.call(shlex.split(f"./vecalign.sh {to_align[s][0]} {to_align[t][0]} {to_align[s][1]} {to_align[s][2]} {to_align[t][1]} {to_align[t][2]} {output_align}"))




traverse(src_path)
for f in folders:
    process_folder(f)
