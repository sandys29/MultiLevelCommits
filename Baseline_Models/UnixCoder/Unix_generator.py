
import torch
import json
import pandas as pd
from transformers import RobertaTokenizer, BertForMaskedLM
from tqdm import tqdm

input_path = "Dataset Path"
model_dir = "Model Path"
output_csv_path = "Path to save the results"

tokenizer = RobertaTokenizer.from_pretrained(model_dir)
model = BertForMaskedLM.from_pretrained(model_dir).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
device = next(model.parameters()).device


def clean_token(token):
    return token.replace("Ġ", " ").strip()


def generate_commit_message(diff, mask_count=10):
    masked_tokens = " ".join([tokenizer.mask_token] * mask_count)
    prompt = f"Generate commit message: {diff} {masked_tokens}"

    for _ in range(mask_count):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        mask_token_index = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)
        if mask_token_index[0].numel() == 0:
            break

        pos = mask_token_index[0][0].item()
        predicted_token_id = logits[0, pos].argmax(dim=-1).item()
        predicted_token = tokenizer.decode([predicted_token_id])
        
        prompt_tokens = prompt.split()
        first_mask_idx = prompt_tokens.index(tokenizer.mask_token)
        prompt_tokens[first_mask_idx] = predicted_token
        prompt = " ".join(prompt_tokens)

    return clean_token(prompt.split("Generate commit message:")[-1].strip())

with open(input_path, "r") as f:
    data = json.load(f)


results = []
for gold_msg, diff in tqdm(data.items(), desc="Generating commit messages"):
    generated_msg = generate_commit_message(diff, mask_count=15)
    results.append({
        "gold_message": gold_msg,
        "generated_message": generated_msg,
        "diff": diff
    })


df = pd.DataFrame(results)
df.to_csv(output_csv_path, index=False)
print(f"\n Saved generated commit messages to: {output_csv_path}")