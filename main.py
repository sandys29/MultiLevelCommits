from Preprocessing import is_whitespace_only_diff, preprocess_total
import json
import argparse
import re
from transformers import AutoTokenizer
from LLMs.GPT import getGPTResponse
from LLMs.Llama import getLlamaResponse
from LLMs.Mistral import getMistralResponse
import csv
import pandas as pd

#Preprocess Total Diffs
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

#Break Total Diff to File-wise Diffs
def split_diff_files(filename):
    new_diff_label=[]
    df = pd.read_csv(f'Outputs/Total_Diff/{filename}.csv')
    df_cleaned = df.replace('ERROR', pd.NA).dropna()
    filtered_diffs = list(df_cleaned['diff'])
    for i in range (0,len(filtered_diffs)):
        result = re.split(r'(?=diff --git)', filtered_diffs[i])
        for res in result:
            if(len(res)>0):
                new_diff_label.append({'diff':res, 'actual_index':i})
    return new_diff_label

if __name__ == '__main__':
    #Get all API Keys
    parser = argparse.ArgumentParser(description='Process necessary API keys.')
    parser.add_argument('--HFKEY', type=str, required=True, help='Required Hugging Face API Key')
    parser.add_argument('--GROQ', type=str, required=True, help='Required Groq API Key')
    parser.add_argument('--OPENAI', type=str, required=True, help='Required OpenAI API Key')
    parser.add_argument('--MISTRAL', type=str, required=True, help='Required Mistral API Key')
    
    #Get File name
    parser.add_argument('--FILE', type=str, required=True, help='Required Input File name')
    parser.add_argument('--OFILE1', type=str, required=True, help='Required Output File name for Total Diffs')
    parser.add_argument('--OFILE2', type=str, required=True, help='Required Output File name for File-wise Diffs')
      
    #Store API keys
    args = parser.parse_args()
    HFKEY = args.HFKEY
    GROQ = args.GROQ
    OPENAI_API_KEY = args.OPENAI
    MISTRAL = args.MISTRAL
    
    #Store File name
    filename = args.FILE
    output_file = args.OFILE1
    output_file_1 = args.OFILE2
    
    #Preprocess File of Total Diffs
    with open(filename, 'r') as file:
        data = file.read()
    diffs_label = json.loads(data)
    filtered_diff_label=preprocess_diffs_label(HFKEY,diffs_label)
    
    print("Generating outputs for total diffs for Multiple LLMs")
    #Initialize containers to store failures
    cf_1, cf_2, cf_3, cf_4 = [], [], [], []
    
    print('Starting Llama3.1 70B')
    #Llama3.1 70B
    cf_1=[]
    for i in range(0, 1):
        if ('llama-70b-output' not in filtered_diff_label[i] or filtered_diff_label[i]['llama-70b-output'] == 'ERROR'):
            diff=filtered_diff_label[i]['diff']
            prompt = f"The following is a diff which describes the code \
                changes in a commit. Your task is to write a short commit \
                message accordingly. {diff} According to the diff, the commit \
                message should be"
            try:
                filtered_diff_label[i]['llama-70b-output'] = getLlamaResponse(GROQ,"llama-3.1-70b-versatile",prompt)
            except:
                cf_1.append(i)
                filtered_diff_label[i]['llama-70b-output'] = 'ERROR'
                pass
    
    print("--Llama3.1 70B model output generated")
    
    #Llama3.1 8B    
    print('Starting Llama3.1 8B')   
    cf_2=[]
    for i in range(0, 1):
        if ('llama3.1-8b-output' not in filtered_diff_label[i] or filtered_diff_label[i]['llama3.1-8b-output'] == 'ERROR'):
            diff=filtered_diff_label[i]['diff']
            prompt = f"The following is a diff which describes the code \
                changes in a commit. Your task is to write a short commit \
                message accordingly. {diff} According to the diff, the commit \
                message should be"
            try:
                filtered_diff_label[i]['llama3.1-8b-output'] = getLlamaResponse(GROQ,"llama-3.1-8b-instant",prompt)
            except:
                cf_2.append(i)
                filtered_diff_label[i]['llama3.1-8b-output'] = 'ERROR'
                pass
    
    print("--Llama3.1 8B model output generated")
    
    #Mistral-Large  
    print('Starting Mistral Large')
    cf_3=[]
    for i in range(0, 1):
        if ('mistral-large-output' not in filtered_diff_label[i] or filtered_diff_label[i]['mistral-large-output'] == 'ERROR'):
            diff=filtered_diff_label[i]['diff']
            prompt = f"The following is a diff which describes the code \
                changes in a commit. Your task is to write a short commit \
                message accordingly. {diff} According to the diff, the commit \
                message should be"
            try:
                filtered_diff_label[i]['mistral-large-output'] = getMistralResponse(MISTRAL,prompt)
            except:
                cf_3.append(i)
                filtered_diff_label[i]['mistral-large-output'] = 'ERROR'
                pass
    
    print("--Mistral Large model output generated")
    
    #GPT-4o
    print('Starting GPT-4o')
    cf_4 = []
    for i in range(0, 1):
        if('gpt-4o-output' not in filtered_diff_label[i] or filtered_diff_label[i]['gpt-4o-output'] == 'ERROR'):
            diff=filtered_diff_label[i]['diff']
            prompt = f"The following is a diff which describes the code \
                changes in a commit. Your task is to write a short commit \
                message accordingly. {diff} According to the diff, the commit \
                message should be"
            try:
                filtered_diff_label[i]['gpt-4o-output'] = getGPTResponse(OPENAI_API_KEY,prompt)
            except:
                cf_4.append(i)
                filtered_diff_label[i]['gpt-4o-output'] = 'ERROR'
                pass
    
    print("--GPT-4o model output generated")
    
    #Save Outputs for Total Diffs
    print("Saving Outputs.....")
    csv_file_name = f'Outputs/Total_Diff/{output_file}.csv'
    headers = filtered_diff_label[0].keys()
    with open(csv_file_name, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(filtered_diff_label)
        
    print(f'Output of all models saved to Outputs/Total_Diff/{output_file}.csv')

    #Split Total Diffs to File-wise Diffs
    print("Split Total Diffs to individual File-wise diffs....")
    new_diff_label = split_diff_files(output_file)
    
    print("Generating outputs for Individual File-wise diffs for Multiple LLMs")
    #Initialize containers to store failures
    cf1, cf2, cf3, cf4 = [], [], [], []
    
    #Llama3.1-70B
    print('Starting Llama3.1 70B')
    cf1=[]
    for i in range(0, len(new_diff_label)):
        if('llama-70b-output' not in new_diff_label[i] or new_diff_label[i]['llama-70b-output'] == 'ERROR'):
            diff=new_diff_label[i]['diff']
            prompt = f"{diff} Given the following code changes, generate a message summarizing \
                the modifications made in this code."
            try:
                new_diff_label[i]['llama-70b-output'] = getLlamaResponse(GROQ,"llama-3.1-70b-versatile",prompt)
            except:
                cf1.append(i)
                new_diff_label[i]['llama-70b-output'] = 'ERROR'
                pass
    print("--Llama3.1 70B model output generated")
    
    #Llama3.1-8B
    print('Starting Llama3.1 8B')
    cf2=[]
    for i in range(0, len(new_diff_label)):
        if('llama3.1-8b-output' not in new_diff_label[i] or new_diff_label[i]['llama3.1-8b-output'] == 'ERROR'):
            diff=new_diff_label[i]['diff']
            prompt = f"{diff} Given the following code changes, generate a message summarizing \
                the modifications made in this code."
            try:
                new_diff_label[i]['llama3.1-8b-output'] = getLlamaResponse(GROQ,"llama-3.1-8b-instant",prompt)
            except:
                cf2.append(i)
                new_diff_label[i]['llama3.1-8b-output'] = 'ERROR'
                pass
    print("--Llama3.1 8B model output generated")

    #Mistral-Large
    print('Starting Mistral Large')
    cf3=[]
    for i in range (0, len(new_diff_label)):
        if('mistral-large-output' not in new_diff_label[i] or new_diff_label[i]['mistral-large-output'] == 'ERROR'):
            diff=new_diff_label[i]['diff']
            prompt = f"{diff} Given the following code changes, generate a message summarizing \
                the modifications made in this code."
            try:
                new_diff_label[i]['mistral-large-output'] = getMistralResponse(MISTRAL,prompt)
            except:
                cf3.append(i)
                new_diff_label[i]['mistral-large-output'] = 'ERROR'
                pass
    print("--Mistral Large model output generated")
    
    #GPT-4o
    print('Starting GPT-4o')
    cf4=[]
    for i in range(0, len(new_diff_label)):
        if('gpt-4o-output' not in new_diff_label[i] or new_diff_label[i]['gpt-4o-output'] == 'ERROR'):
            diff=new_diff_label[i]['diff']
            prompt = f"{diff} Given the following code changes, generate a message summarizing \
                the modifications made in this code."
            try:
                new_diff_label[i]['gpt-4o-output'] = getGPTResponse(OPENAI_API_KEY,prompt)
            except:
                cf4.append(i)
                new_diff_label[i]['gpt-4o-output'] = 'ERROR'
                pass
    print("--GPT-4o model output generated")
    
    #Save outputs for File wise Diffs
    csv_file_name = f'Outputs/File_Level_Diff/{output_file_1}.csv'
    headers = new_diff_label[0].keys()
    with open(csv_file_name, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(new_diff_label)
        
    print(f'Output of all models saved to Outputs/File_Level_Diff/{output_file_1}.csv')