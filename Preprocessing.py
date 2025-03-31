from transformers import AutoTokenizer
import re

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