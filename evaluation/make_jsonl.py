""" Simple script to make a jsonl file given system hypotheses for metricX scoring """
import json
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--hyp", type = str, help = "Path to hypotheses")
    parser.add_argument("--out", type = str, help = "Output path")

    return parser.parse_args()

def main():
    args = parse_args()
    with open("references.txt", "r", encoding = "utf-8") as ref:
        ref_lines = ref.read().strip().split("\n")
    with open(args.hyp, "r", encoding = "utf-8") as hyp:
        hyp_lines = hyp.read().strip().split("\n")

    with open(args.out, "w", encoding = "utf-8") as out:
        for hyp, ref in zip(hyp_lines, ref_lines):
            dict = {"reference": ref.strip(), "hypothesis": hyp.strip()}
            json_string = json.dumps(dict, ensure_ascii = False)
            out.write(f"{json_string}\n")

if __name__ == "__main__":
    main()
