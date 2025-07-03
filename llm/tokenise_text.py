#0. INITIAL SET-UP --------------------------------------------------------------------------

#import libraries
from transformers import AutoTokenizer
import json
import time
from rich.progress import track
import evaluate
import unicodedata
import re

#set params
data_path = "llm/data/madlad_from_huggingface/gd_clean_0000.json"
model_name = "meta-llama/Llama-3.2-1B"


#1. LOAD DATA AND TOKENISE --------------------------------------------------------------------------

#load data
for item in track(range(1), description = 'Loading data...'):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

texts = [item.get('text', '') for item in data[:10]]

#tokenise data
for item in track(range(1), description = 'Tokenising data...'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_text = [tokenizer.tokenize(text) for text in texts]

#remove gdot token
tokenized_text = [[token.replace('Ġ', '') for token in tokens] for tokens in tokenized_text]

def replace_a_tilde_utf8_tokens(tokens):
    replaced_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if 'Ã' in token:
            index = token.index('Ã')

            # Case 1: Next character is in the same token
            if index + 1 < len(token):
                a_tilde_pair = token[index:index+2]
                try:
                    decoded = a_tilde_pair.encode('latin-1').decode('utf-8')
                    new_token = token[:index] + decoded + token[index+2:]
                    replaced_tokens.append(new_token)
                except (UnicodeDecodeError, UnicodeEncodeError):
                    replaced_tokens.append(token)  # Fallback
            # Case 2: Next character is in the next token
            elif i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token:  # ensure not empty
                    pair = 'Ã' + next_token[0]
                    try:
                        decoded = pair.encode('latin-1').decode('utf-8')
                        new_token = token.replace('Ã', decoded, 1)
                        next_token_remainder = next_token[1:]
                        replaced_tokens.append(new_token)
                        if next_token_remainder:
                            replaced_tokens.append(next_token_remainder)
                        i += 1  # Skip the next token since we used part of it
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        replaced_tokens.append(token)
                else:
                    replaced_tokens.append(token)
            else:
                replaced_tokens.append(token)
        else:
            replaced_tokens.append(token)

        i += 1

    return replaced_tokens

tokenized_text_fixed = [replace_a_tilde_utf8_tokens(tokens) for tokens in tokenized_text]
