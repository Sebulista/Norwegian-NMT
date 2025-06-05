#!/bin/bash

# This script outputs all the sentence pairs that were filtered out due to the
# 'no_identical' rule.

# Define the base directory
BASE_DIR="path/to/source/directory"

# Loop through each subdirectory in the base directory
for SUB_DIR in "$BASE_DIR"/*/; do
    # Check if it is indeed a directory
    if [ -d "$SUB_DIR" ]; then
        # Navigate to the subdirectory
        cd "$SUB_DIR" || exit

        echo -e "\nMoved to $SUB_DIR"
        
        # Iterate over files within the subdirectory
        for FILE in *; do
            # Check if it is a file
            suffix="$(basename "$SUB_DIR")"
            output_file="identical.$suffix"
            if [ -f "$FILE" ] && [ "$FILE" != "$output_file" ] ; then
                # Run the command on the file
                echo "Processing $FILE"
                cat "$FILE" | grep -P "\tno_identical" >> "$output_file"
            fi
        done
        
        # Navigate back to the base directory
        cd "$BASE_DIR" || exit
    fi
done
