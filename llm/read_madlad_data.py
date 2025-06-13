#import libraries
import gzip
import json

#set data paths
madlad_data_path_zip_clean = "llm/data/madlad_from_huggingface/gd_clean_0000.jsonl.gz"
madlad_data_path_zip_noisy = "llm/data/madlad_from_huggingface/gd_clean_0000.jsonl.gz"
madlad_data_path_json_clean = "llm/data/madlad_from_huggingface/gd_clean_0000.json"
madlad_data_path_json_noisy = "llm/data/madlad_from_huggingface/gd_noisy_0000.json"


#READ ORIGINAL FILE ------------------------------------------------------------------------------------

# #CLEAN ----
# #open original text file
# with gzip.open(madlad_data_path_zip_clean, "rt", encoding="utf-8") as f:
#     data = [json.loads(line) for line in f]

# #print sentences
# for i, item in enumerate(data[:20], 1):
#     print(f"{i}: {item['text']}\n---")

# #write to a json file
# with open("madlad_data_path_json_clean, "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=2) #ensure_ascii=False preserves Gaelic accents

# #NOISY ----
# #open original text file
# with gzip.open(madlad_data_path_zip_noisy, "rt", encoding="utf-8") as f:
#     data = [json.loads(line) for line in f]

# #print sentences
# for i, item in enumerate(data[:20], 1):
#     print(f"{i}: {item['text']}\n---")

# #write to a json file
# with open(madlad_data_path_json_noisy, "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=2) #ensure_ascii=False preserves Gaelic accents


#READ JSON VERSION --------------------------------------------------------------------------------------

#CLEAN ----
#open json file
with open(madlad_data_path_json_clean, "r", encoding="utf-8") as f:
    data = json.load(f)

#get number of sentences in the data 
num_items = len(data)
print(f"Number of items (clean): {num_items}")

#get approx tokens in the data
num_tokens = sum(len(item.get("text", "").split()) for item in data)
print(f"Number of space-split tokens (clean): {num_tokens}")

#print first few lines
for i, item in enumerate(data[:10], 1):
    print(f"{i}: {item.get('text', '')}\n---")

#NOISY ----
#open json file
with open(madlad_data_path_json_noisy, "r", encoding="utf-8") as f:
    data = json.load(f)

#get number of sentences in the data 
num_items = len(data)
print(f"Number of items (noisy): {num_items}")

#get approx tokens in the data
num_tokens = sum(len(item.get("text", "").split()) for item in data)
print(f"Number of space-split tokens (noisy): {num_tokens}")

#print first few lines
for i, item in enumerate(data[:10], 1):
    print(f"{i}: {item.get('text', '')}\n---")