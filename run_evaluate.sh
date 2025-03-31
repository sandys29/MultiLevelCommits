#!/bin/bash

# Predefined location of the files
FILE_PATH="Outputs/Total_Diff/"

# Check if exactly 5 arguments (file names) are provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 file1 file2 file3 file4 file5"
    exit 1
fi

# Assign arguments (file names) to variables
FILE1="$FILE_PATH$1"
FILE2="$FILE_PATH$2"
FILE3="$FILE_PATH$3"
FILE4="$FILE_PATH$4"
FILE5="$FILE_PATH$5"

# Run evaluate.py with the file names and their paths
python3 Evaluation/Automatic_Evaluation/evaluate.py --FILE1 "$FILE1" --FILE2 "$FILE2" --FILE3 "$FILE3" --FILE4 "$FILE4" --FILE5 "$FILE5"
