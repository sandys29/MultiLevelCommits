#!/bin/bash

# Check if all required arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 HFKEY OPENAI"
    exit 1
fi

# Assign arguments to variables
HFKEY=$1
OPENAI=$2

# Run main.py with the passed arguments
python3 main.py --HFKEY "$HFKEY" --OPENAI "$OPENAI" --FILE "sample_200/1k_msg_rawdiff_part_1.txt" --OFILE "diff_output_1"
python3 main.py --HFKEY "$HFKEY" --OPENAI "$OPENAI" --FILE "sample_200/1k_msg_rawdiff_part_2.txt" --OFILE "diff_output_2"
python3 main.py --HFKEY "$HFKEY" --OPENAI "$OPENAI" --FILE "sample_200/1k_msg_rawdiff_part_3.txt" --OFILE "diff_output_3"
python3 main.py --HFKEY "$HFKEY" --OPENAI "$OPENAI" --FILE "sample_200/1k_msg_rawdiff_part_4.txt" --OFILE "diff_output_4"
python3 main.py --HFKEY "$HFKEY" --OPENAI "$OPENAI" --FILE "sample_200/1k_msg_rawdiff_part_5.txt" --OFILE "diff_output_5"