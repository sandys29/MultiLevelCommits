#!/bin/bash

# Check if all required arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 HFKEY GROQ OPENAI MISTRAL"
    exit 1
fi

# Assign arguments to variables
HFKEY=$1
GROQ=$2
OPENAI=$3
MISTRAL=$4

# Run main.py with the passed arguments
python3 main.py --HFKEY "$HFKEY" --GROQ "$GROQ" --OPENAI "$OPENAI" --MISTRAL "$MISTRAL" --FILE "sample_200/1k_msg_rawdiff_part_1.txt" --OFILE1 "diff_output_1" --OFILE2 "split_diff_output_1"
python3 main.py --HFKEY "$HFKEY" --GROQ "$GROQ" --OPENAI "$OPENAI" --MISTRAL "$MISTRAL" --FILE "sample_200/1k_msg_rawdiff_part_2.txt" --OFILE1 "diff_output_2" --OFILE2 "split_diff_output_2"
python3 main.py --HFKEY "$HFKEY" --GROQ "$GROQ" --OPENAI "$OPENAI" --MISTRAL "$MISTRAL" --FILE "sample_200/1k_msg_rawdiff_part_3.txt" --OFILE1 "diff_output_3" --OFILE2 "split_diff_output_3"
python3 main.py --HFKEY "$HFKEY" --GROQ "$GROQ" --OPENAI "$OPENAI" --MISTRAL "$MISTRAL" --FILE "sample_200/1k_msg_rawdiff_part_4.txt" --OFILE1 "diff_output_4" --OFILE2 "split_diff_output_4"
python3 main.py --HFKEY "$HFKEY" --GROQ "$GROQ" --OPENAI "$OPENAI" --MISTRAL "$MISTRAL" --FILE "sample_200/1k_msg_rawdiff_part_5.txt" --OFILE1 "diff_output_5" --OFILE2 "split_diff_output_5"