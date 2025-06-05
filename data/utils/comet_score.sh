#!/bin/bash

# Usage: ./count_scores.sh THRESHOLD DIRECTORY DELIMITER
# Example: ./count_scores.sh 0.5 /path/to/directory "," 
# Function: Calculate and append the cometkiwi score to a folder
# 	    of bitext files

DIRECTORY=$1
OUTPUT_DIR=$2
START_INDEX=$3
N=$4
#DELIMITER=${3:-"\t"}  # Default delimiter is tab if not specified
DELIMITER='\t'

if [ -z "$DIRECTORY" ]; then
    echo "Please provide a directory containing the files to process."
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Please provide an output directory."
    exit 1
fi

# Check if the provided directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Directory $DIRECTORY does not exist."
    exit 1
fi


# Iterate over each text file in the directory
#for file in "$DIRECTORY"/*; do

#find and sort the files
FILES=$(find "$DIRECTORY" -type f -name "*" -printf "%s %p\n" | sort -n)

#skip the files up to start_index
#SELECTED_FILES=$(echo "$FILES" | tail -n +$((START_INDEX + 1)) | head -n "$N"| cut -d' ' -f2-)

for file in $(find "$DIRECTORY" -type f -name "*" -printf "%s %p\n" | sort -n | cut -d ' ' -f2 | tail -n +$((START_INDEX)) | head -n "$N"); do
    if [ -f "$file" ]; then
        echo "File $file"
        filename=$(basename "$file")
        output_file="$OUTPUT_DIR/$filename"
        paste $file <(cut -f3 <(comet-score -s <(cut -f1 $file) -t <(cut -f2 $file) --quiet --batch_size 32 --model path/to/comet-models/wmt22-cometkiwi-da/checkpoints/model.ckpt) | cut -d ' ' -f2) > $output_file
    fi
done
