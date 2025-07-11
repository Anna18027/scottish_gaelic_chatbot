
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "goldfish-models/eng_latn_1000mb"
# model_name = "goldfish-models/ace_latn_10mb"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#set generation params
max_new_tokens = 30
temperature = 0.8
top_p = 0.9
do_sample = True


model.to("cpu")
model.eval()

#set up Gaelic and English prompts
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

print("\n--- Sentence Completions ---\n")

gaelic_completions = []
english_completions = []

for i, (gaelic_prompt, english_prompt) in enumerate(zip(prompts, english_prompts)):
    print(f"[{i}] GAELIC PROMPT:  {gaelic_prompt}")
    print(f"    ENGLISH PROMPT: {english_prompt}")

    print("  [DEBUG] Gaelic Tokens:", tokenizer.tokenize(gaelic_prompt))
    print("  [DEBUG] English Tokens:", tokenizer.tokenize(english_prompt))


    # Gaelic prompt completion
    input_ids_gaelic = tokenizer(gaelic_prompt, return_tensors="pt").input_ids.to("cpu")
    with torch.no_grad():
        output_gaelic = model.generate(
            input_ids=input_ids_gaelic,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded_gaelic = tokenizer.decode(output_gaelic[0], skip_special_tokens=True)
    completion_gaelic = decoded_gaelic[len(gaelic_prompt):].strip()
    print(f"    [GAELIC COMPLETION]:  {completion_gaelic}")

    # English prompt completion
    input_ids_english = tokenizer(english_prompt, return_tensors="pt").input_ids.to("cpu")
    with torch.no_grad():
        output_english = model.generate(
            input_ids=input_ids_english,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded_english = tokenizer.decode(output_english[0], skip_special_tokens=True)
    completion_english = decoded_english[len(english_prompt):].strip()
    print(f"    [ENGLISH COMPLETION]: {completion_english}\n")

    # Save each
    gaelic_completions.append({
        "prompt": gaelic_prompt,
        "completion": completion_gaelic
    })

    english_completions.append({
        "prompt": english_prompt,
        "completion": completion_english
    })