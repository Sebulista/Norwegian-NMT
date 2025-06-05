#!/bin/bash
# A simple script to sort each sentence pair in a file by the score column
# (the third column in the file)

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 path/to/dir"
    exit 1
fi

dir_path="$1"

if [ -d "$dir_path" ]; then
    for file in "$dir_path"/*; do
        if [ -f "$file" ]; then
            echo Processing "$file"
            sort -t$'\t' -k3,3nr "$file" -o "$file.sorted"
            mv "$file.sorted" "$file"
        fi
    done
fi
