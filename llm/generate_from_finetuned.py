#load libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#set generation params
max_new_tokens = 30
temperature = 0.8
top_p = 0.95
do_sample = True

#set file paths
finetuned_path = "llm/model_results/llama_finetuned"
base_model_id = "timinar/baby-llama-58m"

#load fine-tuned model
finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_path)
finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
finetuned_model.to("cpu")
finetuned_model.eval()

#load base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
base_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model.to("cpu")
base_model.eval()

#set up Gaelic half sentences as prompts
prompts = [
    "Tha mi a’ smaoineachadh gu bheil e",
    "Chuala mi an naidheachd",
    "Bha an latha ro",
    "Thàinig iad gu",
    "Tha mi a’ dol a",
]

english_prompts = [
    "I think that it is",
    "I heard the news",
    "The day was too",
    "They came to",
    "I am going to",
]


#print results
print("\n--- Sentence Completions Comparison ---\n")

for i, prompt in enumerate(prompts):
    print(f"[{i}] PROMPT: {prompt}")

    # Fine-tuned model completion
    input_ids_finetuned = finetuned_tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")
    with torch.no_grad():
        output_finetuned = finetuned_model.generate(
            input_ids=input_ids_finetuned,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=finetuned_tokenizer.pad_token_id,
            eos_token_id=finetuned_tokenizer.eos_token_id,
        )
    decoded_finetuned = finetuned_tokenizer.decode(output_finetuned[0], skip_special_tokens=True)
    generated_finetuned = decoded_finetuned[len(prompt):].strip()
    print(f"    [Finetuned] COMPLETION: {generated_finetuned}")

    # Base model completion
    input_ids_base = base_tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")
    with torch.no_grad():
        output_base = base_model.generate(
            input_ids=input_ids_base,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=base_tokenizer.pad_token_id,
            eos_token_id=base_tokenizer.eos_token_id,
        )
    decoded_base = base_tokenizer.decode(output_base[0], skip_special_tokens=True)
    generated_base = decoded_base[len(prompt):].strip()
    print(f"    [Original]  COMPLETION: {generated_base}\n")
