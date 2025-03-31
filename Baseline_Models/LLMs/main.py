from GPT3_5 import generate_commit_message_gpt
from llama2 import generate_commit_message_llama
import json
import argparse
import re
from transformers import AutoTokenizer
from tqdm import tqdm
import csv

def is_whitespace_only_diff(diff):
    lines = diff.splitlines()
    added_pattern = re.compile(r'^\+')
    removed_pattern = re.compile(r'^\-')
    for line in lines:
        if added_pattern.match(line) or removed_pattern.match(line):
            # Remove the initial '+' or '-' and compare the content ignoring whitespace
            content = line[1:].strip()
            if content:
                return False
    return True

def preprocess_total(diffs_label):
    labels = list(diffs_label.keys())
    diffs = list(diffs_label.values())
    filtered_diff_label = []
    HF_TOKEN = 'Hugging_Face_API_Key'
    pattern = re.compile(r'\d+ \. \d+( \. \d+)*')
    filtered_labels, filtered_diffs = [],[]
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", token = HF_TOKEN)
    for i in range(0,len(labels)):
        n_tokens = len(tokenizer.encode(diffs[i]))
        if (not pattern.search(labels[i])) and (not is_whitespace_only_diff(diffs[i])) and n_tokens<131072:
            filtered_labels.append(labels[i])
            filtered_diffs.append(diffs[i])
            filtered_diff_label.append({'diff':diffs[i], 'label':labels[i]})
    return filtered_diff_label

def preprocess_diffs_label(HFKEY, diffs_label):
    labels = list(diffs_label.keys())
    diffs = list(diffs_label.values())
    filtered_diff_label = []
    pattern = re.compile(r'\d+ \. \d+( \. \d+)*')
    filtered_labels, filtered_diffs = [],[]
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", token = HFKEY)
    for i in range(0,len(labels)):
        n_tokens = len(tokenizer.encode(diffs[i]))
        if (not pattern.search(labels[i])) and (not is_whitespace_only_diff(diffs[i])) and n_tokens<131072:
            filtered_labels.append(labels[i])
            filtered_diffs.append(diffs[i])
            filtered_diff_label.append({'diff':diffs[i], 'label':labels[i]})
    return filtered_diff_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process necessary API keys.')
    parser.add_argument('--HFKEY', type=str, required=True, help='Required Hugging Face API Key')
    parser.add_argument('--OPENAI', type=str, required=True, help='Required OpenAI API Key')
    
    #Get File name
    parser.add_argument('--FILE', type=str, required=True, help='Required Input File name')
    parser.add_argument('--OFILE', type=str, required=True, help='Required Output File name for Total Diffs')
    
    #Store API keys
    args = parser.parse_args()
    HFKEY = args.HFKEY
    OPENAI_API_KEY = args.OPENAI
    
    #Store File name
    filename = args.FILE
    output_file = args.OFILE
    
    with open(filename, 'r') as file:
        data = file.read()
    diffs_label = json.loads(data)
    filtered_diff_label=preprocess_diffs_label(HFKEY,diffs_label)
    
    cf_1 = []
    for i in tqdm(range(0, len(filtered_diff_label))):
        if('gpt-3.5-output' not in filtered_diff_label[i] or filtered_diff_label[i]['gpt-3.5-output'] == 'ERROR'):
            diff=filtered_diff_label[i]['diff']
            prompt = f"The following is a diff which describes the code \
                changes in a commit. Your task is to write a short commit \
                message accordingly. {diff} According to the diff, the commit \
                message should be"
            try:
                filtered_diff_label[i]['gpt-3.5-output'] = generate_commit_message_gpt(OPENAI_API_KEY,prompt)
            except:
                cf_1.append(i)
                filtered_diff_label[i]['gpt-3.5-output'] = 'ERROR'
                pass
    
    cf_2 = []
    for i in tqdm(range(0, len(filtered_diff_label))):
        if('llama2-Output' not in filtered_diff_label[i] or filtered_diff_label[i]['llama2-Output'] == 'ERROR'):
            diff=filtered_diff_label[i]['diff']
            try:
                filtered_diff_label[i]['llama2-Output'] = generate_commit_message_llama(diff)
            except:
                cf_2.append(i)
                filtered_diff_label[i]['llama2-Output'] = 'ERROR'
                pass
    
    print("Saving Outputs.....")
    csv_file_name = f'Output.csv'
    headers = filtered_diff_label[0].keys()
    with open(csv_file_name, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(filtered_diff_label)
        
    print(f'Output of all models saved to Output.csv')