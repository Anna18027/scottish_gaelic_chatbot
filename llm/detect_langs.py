#0. INITIAL SET-UP --------------------------------------------------------------------------

#import libraries
from transformers import AutoTokenizer
import json
import time
from rich.progress import track
import evaluate
import unicodedata
import re
from fast_langdetect import detect
from collections import Counter

#set params
data_path = "llm/data/madlad_from_huggingface/gd_clean_0000.json"
model_name = "meta-llama/Llama-3.2-1B"
token_output_path = "llm/data/madlad_from_huggingface/gd_clean_0000_ws_tokenized.json"


#1. LOAD DATA AND TOKENISE --------------------------------------------------------------------------

# #load data
# for item in track(range(1), description = 'Loading data...'):
#     with open(data_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

# texts = [item.get('text', '') for item in data[:10]]

# def simple_tokenize(text):
#     return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

# #tokenise data
# for item in track(range(1), description = 'Tokenising data...'):
#     tokenized_text = [simple_tokenize(text) for text in texts]

# # SAVE tokenized output
# with open(token_output_path, "w", encoding="utf-8") as f:
#     json.dump(tokenized_text, f, ensure_ascii=False, indent=2)

# LOAD tokenized output again
with open(token_output_path, "r", encoding="utf-8") as f:
    tokenized_text = json.load(f)


#2. DETECT LANGUAGE --------------------------------------------------------------------------------

def detect_languages_for_tokens(token_lists):
    token_langs_per_sentence = []
    all_token_langs = []

    for tokens in token_lists:
        token_langs = []
        for token in tokens:
            try:
                detected = detect(token)
                if isinstance(detected, dict) and 'lang' in detected:
                    lang = detected['lang']
                else:
                    lang = detected
            except Exception:
                lang = "unknown"
            token_langs.append((token, lang))
            all_token_langs.append(lang)

        token_langs_per_sentence.append(token_langs)

    return token_langs_per_sentence, all_token_langs

token_langs_per_sentence, all_token_langs = detect_languages_for_tokens(tokenized_text)

# -----------------------
# Print per-token languages
# -----------------------
for i, sentence_langs in enumerate(token_langs_per_sentence):
    print(f"\nSentence {i + 1}:")
    for token, lang in sentence_langs:
        print(f"  {token:15} → {lang}")

# -----------------------
# Per-sentence proportions
# -----------------------
print("\n--- Language proportions per sentence ---")
for i, sentence_langs in enumerate(token_langs_per_sentence):
    langs = [lang for _, lang in sentence_langs]
    counts = Counter(langs)
    total = sum(counts.values())
    proportions = {lang: f"{count / total:.2%}" for lang, count in counts.items()}
    print(f"Sentence {i + 1}: {proportions}")

# -----------------------
# Overall proportions
# -----------------------
print("\n--- Overall language proportions ---")
overall_counts = Counter(all_token_langs)
total_tokens = sum(overall_counts.values())
overall_proportions = {lang: f"{count / total_tokens:.2%}" for lang, count in overall_counts.items()}
print(overall_proportions)



#detect language by token


#detect language by sentence


#display results

# text = "C'était déjà l'été à São Paulo agus ciamar a tha thu."

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Byte-level tokens (LLaMA tokenizer)
# token_ids = tokenizer.encode(text, add_special_tokens=False)
# tokens = tokenizer.convert_ids_to_tokens(token_ids)
# print(tokens)
# # Output might be something like: ['C', "'", 'Ã', '©', 't', 'ait', '...', 'Ã', '£', '...']

# # Decoded whole text (human-readable)
# decoded_text = tokenizer.decode(token_ids)
# print(decoded_text)

# word_tokens = re.findall(r"\w+['’]?\w*|[^\w\s]", text, flags=re.UNICODE)
# print(word_tokens)


import matplotlib.pyplot as plt

# Assuming token_langs_per_sentence is a list of lists like:
# [[(token1, lang1), (token2, lang2), ...], [...], ...]

gaelic_code = "gd"  # or whatever your detector uses for Gaelic
english_code = "en"

# Collect per-sentence proportions
gaelic_props = []
english_props = []

for sentence_langs in token_langs_per_sentence:
    langs = [lang for _, lang in sentence_langs]
    total = len(langs)
    if total == 0:
        gaelic_props.append(0)
        english_props.append(0)
        continue
    
    gaelic_count = sum(1 for l in langs if l == gaelic_code)
    english_count = sum(1 for l in langs if l == english_code)

    gaelic_props.append(gaelic_count / total)
    english_props.append(english_count / total)

# Plotting
sentences = list(range(1, len(token_langs_per_sentence) + 1))

plt.figure(figsize=(10, 6))
plt.plot(sentences, gaelic_props, label="Gaelic (gd)", marker='o')
plt.plot(sentences, english_props, label="English (en)", marker='o')
plt.xlabel("Sentence Number")
plt.ylabel("Proportion")
plt.title("Gaelic vs English Proportions per List Item (using tokens)")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.show()


import spacy

nlp = spacy.load("en_core_web_sm")

data_path = "llm/data/madlad_from_huggingface/gd_clean_0000.json"

# Load data
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

gaelic_code = "gd"
english_code = "en"

# def detect_lang_sentence(sentence):
#     try:
#         return detect(sentence)
#     except Exception:
#         return "unknown"

# For each list item, split into sentences, detect lang per sentence, calculate proportions
gaelic_props = []
english_props = []

def sent_tokenize(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

for item in data[:10]:  # or full data if you want
    text = item.get('text', '')
    sentences = sent_tokenize(text)

    langs = [detect(sent) for sent in sentences]
    total = len(langs)
    if total == 0:
        gaelic_props.append(0)
        english_props.append(0)
        continue

    print(f"langs: {langs}")

    gaelic_count = sum(1 for l in langs if l['lang'] == gaelic_code)
    english_count = sum(1 for l in langs if l['lang'] == english_code)

    print(f"gaelic_count: {gaelic_count}")
    print(f"english_count: {english_count}")

    gaelic_props.append(gaelic_count / total)
    english_props.append(english_count / total)

print(f"gaelic_props: {gaelic_props}")
print(f"english_props: {english_props}")

# Plot per list item proportions
items = list(range(1, len(gaelic_props) + 1))

plt.figure(figsize=(10, 6))
plt.plot(items, gaelic_props, label="Gaelic (gd)", marker='o')
plt.plot(items, english_props, label="English (en)", marker='o')
plt.xlabel("List Item Number")
plt.ylabel("Proportion of Sentences")
plt.title("Gaelic vs English Sentence Proportions per List Item (using sentences)")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.show()
