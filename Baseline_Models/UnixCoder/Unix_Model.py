import os
import json
import torch
from datasets import Dataset
from transformers import RobertaTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

os.environ["USE_TF"] = "0"


data_path = "Dataset_path"
model_name = "microsoft/unixcoder-base"
output_dir = "Path to save the model"


tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)


with open(data_path, "r") as f:
    raw_data = json.load(f)


data_list = []
for msg, diff in raw_data.items():
    msg_tokens = tokenizer.tokenize(msg)
    mask_count = min(len(msg_tokens), 20)
    masked_tokens = " ".join(["[MASK]"] * mask_count)
    input_text = f"Commit message for the following diff: {diff} {masked_tokens} </s>"
    data_list.append({"text": input_text})


dataset = Dataset.from_list(data_list)


def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )


tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])


training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=8,         
    per_device_eval_batch_size=8,
    num_train_epochs=5,                   
    learning_rate=5e-5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir=os.path.join(output_dir, "logs"),
    fp16=True,
    save_total_limit=2,
    logging_steps=20,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.3
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()


model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\n Fine-tuned UniXcoder v2 (MLM) saved to: {output_dir}")


