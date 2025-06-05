"""CLI program to join aligned monolingual files into bitext """
from argparse import ArgumentParser
import sys

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--source", "-s", type = str, help = "Path to monolingual source file")
    parser.add_argument("--target", "-t", type = str, help = "Path to monolingual target file")
    #parser.add_argument("--alignment", "-a", type = str, help = "Path to alignment file")
    #parser.add_argument("--output", "-o", type = str, help = "Path to output file")

    return parser.parse_args()



def align(args):
    #with open(args.alignment, "r", encoding = "utf-8") as afile:
    #    alignments = afile.readlines()

    map = []
    for line in sys.stdin:
        if line == "":
            continue
        sources, targets = line.split(":")[:2]
        try:
            sources = [int(s.strip()) for s in sources[1:-1].split(",")]
            targets = [int(t.strip()) for t in targets[1:-1].split(",")]

            map.append((sources, targets))
        except ValueError:
            #print(f"Skipping empty line: {sources} or {targets}")
            pass

    with open(args.source, "r", encoding = "utf-8") as sfile:
        source = sfile.read().strip().split("\n")

    with open(args.target, "r", encoding = "utf-8") as tfile:
        target = tfile.read().strip().split("\n")

    new_lines = []

    for alignment in map:
        source_line = ""
        for src_i in alignment[0]:
            source_line += source[src_i] + " "
        
        target_line = ""
        for tgt_i in alignment[1]:
            target_line += target[tgt_i] + " "

        source_line.strip().replace("\t","")
        target_line.strip().replace("\t","")

        new_lines.append(f"{source_line}\t{target_line}\n")


    #with open(args.output, "w", encoding = "utf-8") as ofile:
    for line in new_lines:
        sys.stdout.write(line)
    


def main():
    args = parse_arguments()
    align(args)

if __name__ == "__main__":
    main()
