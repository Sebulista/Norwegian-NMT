""" Simple script to get metricX score from jsonl file """
import json
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", type = str, help = "Path to input jsonl")

    return parser.parse_args()


def main():
    args = parse_args()
    
    with open(args.input, "r", encoding = "utf-8") as infile:
        lines = infile.read().strip().split("\n")

    sum_x = 0.0
    
    for line in lines:
        data = json.loads(line)
        sum_x += data["prediction"]

    print(sum_x / len(lines))
    print(f"{(sum_x / len(lines)):.2%}")

if __name__ == "__main__":
    main()
