from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

def generate_commit_message_llama(diff_text):
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    prompt = f"""<s>[INST] Generate a meaningful git commit message from the following diff:\n{diff_text} [/INST]"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )

    return tokenizer.decode(output[0], skip_special_tokens=True).strip()