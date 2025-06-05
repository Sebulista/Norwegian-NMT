""" This script performs bicleaner hardrules on the giellatekno data
Remember to set all paths, language directions are hard-coded.
bicleaner-hardrules needs to be installed"""

from pathlib import Path
import subprocess
import shlex

src_path = Path("path/to/tsv_data")
out_path = Path("path/to/output_folder")
bicleaner_path = Path("path/to/bicleaner_models")

folders = []

def traverse(source):
    for child in source.iterdir():
        if child.is_dir():
            traverse(child)
        else:
            folders.append(child.parent)
            break


def process_folder(folder_source_path):
    for child in folder_source_path.iterdir():
        assert(child.is_file())
        direction = child.name.split(".")[-1]
        print(direction)

if __name__ == "__main__":
    traverse(src_path)

    # Holds the tab separated lines that were not discarded
    en_nb = open(out_path / "giellatekno.en_nb", "a")
    en_nn = open(out_path / "giellatekno.en_nn", "a")
    nb_nn = open(out_path / "giellatekno.nb_nn", "a")

    # Holds the full annotated line for each sentence pair
    en_nb_annotated = open(out_path / "giellatekno_annotated.en_nb", "a")
    en_nn_annotated = open(out_path / "giellatekno_annotated.en_nn", "a")
    nb_nn_annotated = open(out_path / "giellatekno_annotated.nb_nn", "a")


    for folder in folders:
        for child in folder.iterdir():
            print(f"Processing {child}")
            assert(child.is_file())
            direction = child.name.split(".")[-1]
            
            src, tgt = direction.split("-")
            metadata = bicleaner_path / f"{direction}/metadata.yaml"
            proc = subprocess.run(shlex.split(f"bicleaner-hardrules {child} - --scol 1 --tcol 2 --annotated_output -s {src} -t {tgt} --metadata {metadata} --disable_lang_ident --disable_porn_removal --disable_minimal_length"), stdout=subprocess.PIPE, text = True)

            lines = proc.stdout.strip()
            if lines == "":
                continue
            lines = lines.split("\n")

            if direction == "en-nb":
                for line in lines:

                    if line.strip() == "":
                        continue

                    split = line.split("\t")
                    if split[2] == "1":
                        en_nb.write(f"{split[0]}\t{split[1]}\n")
                    else:
                        en_nb_annotated.write(f"{line}\n")

            elif direction == "en-nn":
                for line in lines:

                    if line.strip() == "":
                        continue

                    split = line.split("\t")
                    if split[2] == "1":
                        en_nn.write(f"{split[0]}\t{split[1]}\n")
                    else:
                        en_nn_annotated.write(f"{line}\n")

            elif direction == "nb-nn":
                for line in lines:

                    if line.strip() == "":
                        continue

                    split = line.split("\t")
                    if split[2] == "1":
                        nb_nn.write(f"{split[0]}\t{split[1]}\n")
                    else:
                        nb_nn_annotated.write(f"{line}\n")



    en_nb.close()
    en_nn.close()
    nb_nn.close()
    
    en_nb_annotated.close()
    en_nn_annotated.close()
    nb_nn_annotated.close()
