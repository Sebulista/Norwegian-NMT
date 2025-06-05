#!/bin/bash

# Usage: ./count_scores.sh THRESHOLD DIRECTORY DELIMITER
# Example: ./count_scores.sh 0.5 /path/to/directory "," 
# Function: Get the number of sentence pairs above score THRESHOLD in a
# 	    directory of bitext files

THRESHOLD=$1
DIRECTORY=$2
#DELIMITER=${3:-"\t"}  # Default delimiter is tab if not specified
DELIMITER='\t'

if [ -z "$THRESHOLD" ]; then
    echo "Please provide a threshold value as the first argument."
    exit 1
fi

if [ -z "$DIRECTORY" ]; then
    echo "Please provide a directory containing the files to process."
    exit 1
fi

# Check if the provided directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Directory $DIRECTORY does not exist."
    exit 1
fi

total_count=0

# Iterate over each text file in the directory
for file in "$DIRECTORY"/*; do
    if [ ! -f "$file" ]; then
        echo "File $file does not exist. Skipping..."
        continue
    fi

    count=$(awk -v threshold="$THRESHOLD" -v FS="$DELIMITER" '$3 > threshold' "$file" | wc -l)
    echo "File $file: $count lines with score above $THRESHOLD"
    total_count=$((total_count + count))
done

echo "Total count: $total_count"
