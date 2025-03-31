# Automatic Commit Message Generation

## Overview
This repository contains the code and resources for generating automated commit messages using state-of-the-art Large Language Models (LLMs), such as GPT-4o, Llama 3.1 (70B and 8B), and Mistral Large. The project addresses two significant challenges in automated commit message generation:

**Limitations of short token lengths:** Many existing approaches rely on datasets with short token lengths, resulting in incomplete or oversimplified commit messages.

**Handling multiple file changes:** Current models often generate a single commit message for multiple file changes, making it difficult for developers to understand specific file modifications.


## Features
- **LLM-powered commit message generation:** Leverages powerful models like GPT-4o, Llama 3.1, and Mistral Large to generate high-quality commit messages for large diffs.
- **Two-level approach:** Generates both a high-level overall commit message and file-specific messages for each file change, providing more granular context for developers.
- **Automatic evaluation metrics:** Uses BLEU, ROUGE, METEOR, and CIDEr metrics to automatically assess the quality of generated commit messages.
- **Human evaluation:** Includes a developer survey to assess the effectiveness of the proposed approach.

## Prerequisites
Before you begin, ensure you have met the following requirements:
- [ ]  You must be running the code in an Unix based OS.
- [ ]  You have downloaded and setup Python version >= 3.10.
- [ ]  You took permission to use Mistral tokenizer in the Hugging Face platform.
- [ ]  You have generated API token from Hugging Face, Groq, OpenAI, Mistral.
- [ ]  You might need to add credits for each platform if needed.

## Run the Code

To Run this project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/sandhyasankar29/File_Commits.git

# Navigate to the project directory
cd File_Commits

#Set up Virtual Environment
python3 -m venv env

#Activate the Environment
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

#Run the models - Replace Tokens with the Actual Data
./run_all.sh Hugging_Face_Token Groq_Token OpenAI_Token Mistral_Token

#To calculate Evaluation Scores
./run_evaluate.sh diff_output_1.csv diff_output_2.csv diff_output_3.csv diff_output_4.csv diff_output_5.csv

```
