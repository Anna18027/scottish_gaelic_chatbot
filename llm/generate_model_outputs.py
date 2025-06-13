
#0. INITIAL SET-UP -----------------------------------------------------------------------------------------------

#import libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import csv
import time
from rich.progress import track
import evaluate

#choose model
# model_name = "goldfish-models/gla_latn_full" 
model_name = "meta-llama/Llama-3.2-1B"
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# model_name = "google/gemma-2-2b"

#set inputs
inputs = ["Tha seo ", 
          "Tha seo às dèidh dhan",
          "Tha seo às dèidh dhan riaghladair Ofcom a ràdh",
          "Tha seo às dèidh dhan riaghladair Ofcom a ràdh nach eil",
          "Tha seo às dèidh dhan riaghladair Ofcom a ràdh nach eil seirbheis a' Phuist Rìoghail air a bhith math gu leòr bho"]

#set other model params
add_bos_token_to_inputs = True
num_tokens_per_output = 20


#1. FUNCTIONS -------------------------------------------------------------------------------------------------

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.float16)
    return model

def generate_lm_output(model, tokenizer, input_text, output_length):
    tokenised_input = tokenizer(input_text, return_tensors="pt")
    output = model.generate(**tokenised_input, max_new_tokens=output_length, do_sample=True, num_beams=1, pad_token_id = tokenizer.pad_token_id)
    decoded_outputs = tokenizer.batch_decode(output, skip_special_tokens=True)
    return decoded_outputs


#2. GENERATE OUTPUTS ------------------------------------------------------------------------------------------------

#tokenise inputs using model tokeniser
for item in track(range(1), description = 'Loading tokenizer...'):
    tokenizer = load_tokenizer(model_name)
    if add_bos_token_to_inputs==True:
        inputs.insert(0, tokenizer.bos_token or "<s>")

#load the model
for item in track(range(1), description = 'Loading model...'):
    model = load_model(model_name)
  
#decode and generate an output for every input prompt
for item in track(range(1), description = 'Generating outputs...'):   
    for i in range(len(inputs)):
        output= generate_lm_output(model, tokenizer, inputs[i], num_tokens_per_output)
        print(f"{i}: {output}\n---")


#3. CALCULATE PERPLEXITY ----------------------------------------------------------------------------------------------

#function to grab sentences from line-separated .txt file
def load_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences

#function to calculate perplexity
def calculate_perplexity(sentences):
    perplexity_metric = evaluate.load("perplexity")
    results = perplexity_metric.compute(predictions=sentences, model_id=model_name)
    return results['mean_perplexity']

#get english test sentences
english_test_file = "llm/data/temp_data/english_test_set.txt"
english_test_sentences = load_sentences(english_test_file)

#get gaidhlig test sentences
gaidhlig_test_file = "llm/data/temp_data/gaidhlig_test_set.txt"
gaidhlig_test_sentences = load_sentences(gaidhlig_test_file)

#get perplexity
english_perplexity = calculate_perplexity(english_test_sentences)
gaidhlig_perplexity = calculate_perplexity(gaidhlig_test_sentences)

print(f"English perplexity: {english_perplexity:.2f}")
print(f"Gaidhlig perplexity: {gaidhlig_perplexity:.2f}")

#get first half of all the sentences
def get_first_half(sentences):
    first_halves = []
    for sentence in sentences:
        words = sentence.strip().split()
        half_length = max(1, len(words) // 2) 
        first_half = ' '.join(words[:half_length])
        first_halves.append(first_half)
    return first_halves

gaidhlig_half_sentences = get_first_half(gaidhlig_test_sentences)

model_generated_gaidhlig = []
for i in range(len(gaidhlig_half_sentences)):
    output = generate_lm_output(model, tokenizer, gaidhlig_half_sentences[i], num_tokens_per_output)
    model_generated_gaidhlig.append(output)

print(model_generated_gaidhlig)

# with open("model_generated_gaidhlig.csv", "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     for item in model_generated_gaidhlig:
#         writer.writerow([item])  