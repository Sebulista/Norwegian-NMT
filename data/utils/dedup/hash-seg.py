#!/usr/bin/env python3
from argparse import ArgumentParser, FileType
from unicodedata import category as cat
from unidecode import unidecode
from xxhash import xxh64
import sys

parser = ArgumentParser()
parser.add_argument('-a', '--aggressive', action='store_true', default=False)
args = parser.parse_args()

# Translate table to remove non alphabetic characters
tbl = [chr(i) for i in range(sys.maxunicode) if not cat(chr(i)).startswith('L')]
remove_non_alpha = str.maketrans('', '', ''.join(tbl))

def main():
    shashes, thashes = set(), set()
    for line in sys.stdin:
        sline = line.rstrip('\n')
        sline2 = line.rstrip('\n')

        if args.aggressive:
            sline2 = unidecode(sline2.lower().translate(remove_non_alpha))

        hash = xxh64(sline2).hexdigest()

        sys.stdout.write(f"{sline}\t{hash}\n")


if __name__ == "__main__":
    main()
