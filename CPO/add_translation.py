""" Simple script to add a 'translation' dict to the json preference dataset
so it is correctly formatted """

from pathlib import Path
import json
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", type = str, help = "Which file to process")
    args = parser.parse_args()

    path = Path(args.file)
    assert path.is_file()
    new_file = path.parent / f"translation_{path.name}"
    
    with open(path, "r", encoding = "utf-8") as in_file, \
        open(new_file, "w", encoding = "utf-8") as out_file:
        data = in_file.read().strip().split("\n")
        for line in data:
            if line == "":
                continue
            dict = json.loads(line)
            new_dict = {"translation": dict}
            json_string = json.dumps(new_dict, ensure_ascii = False)
            out_file.write(f"{json_string}\n")
