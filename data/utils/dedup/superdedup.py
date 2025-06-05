#!/usr/bin/env python3
import sys
import os
import pickle

def main():
    hashes = set()
    # Try to old existing hashes
    if os.path.isfile('hashes.pickle'):
        with open('hashes.pickle', 'rb') as f:
            hashes = pickle.load(f)

    for line in sys.stdin:
        parts = line.rstrip("\n").split('\t')

        hash = parts[2]

        if hash not in hashes:
            sys.stdout.write(line)
        hashes.add(hash)
    # Write a list of seen hashes
    with open('hashes.pickle','wb') as f:
         pickle.dump(hashes, f)

if __name__ == "__main__":
    main()
