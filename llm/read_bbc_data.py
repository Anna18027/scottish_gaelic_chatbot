#set data paths
bbc_data_path = "llm/data/bbc_data_from_ondrej/bbc.txt.normalized_v2"
bbc_data_path_archive = "llm/data/bbc_data_from_ondrej/bbc.txt.normalized_v2"
bbc_data_path_txt = "llm/data/bbc_data_from_ondrej/bbc_normalized_v2.txt"
bbc_data_path_txt_archive = "llm/data/bbc_data_from_ondrej/bbc_archive_normalized_v2.txt"


#READ ORIGINAL FILE ------------------------------------------------------------------------------------

# #BBC ----
# #open original text file
# with open(bbc_data_path, "r", encoding="utf-8") as f:
#     content = f.read()
    
# #split by \n
# lines = content.split('\n')

# #print first few lines
# for line in lines[:4]:
#     print(repr(line)) 

# #save as txt file
# with open(bbc_data_path_txt, "w", encoding="utf-8") as out_file:
#     for line in lines:
#         out_file.write(line + "\n")

# #ARCHIVE ----
# #open original text file
# with open(bbc_data_path_archive, "r", encoding="utf-8") as f:
#     content = f.read()
    
# #split by \n
# lines = content.split('\n')

# #print first few lines
# for line in lines[:4]:
#     print(repr(line)) 

# #save as txt file
# with open(bbc_data_path_txt_archive, "w", encoding="utf-8") as out_file:
#     for line in lines:
#         out_file.write(line + "\n")


#READ TXT VERSION --------------------------------------------------------------------------------------

#BBC ----
#open txt version
with open(bbc_data_path_txt, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

#print first few lines
for i, sentence in enumerate(sentences[:5], 1):
    print(f"{i}: {sentence}\n---")

#check number of sentences and approx tokens
num_sentences = len(sentences)
num_tokens = sum(len(sentence.split()) for sentence in sentences)

print(f"Number of sentences (bbc): {num_sentences}")
print(f"Number of space-split tokens (bbc): {num_tokens}")

#ARCHIVE ----
#open txt version
with open(bbc_data_path_txt_archive, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

#print first few lines
for i, sentence in enumerate(sentences[:5], 1):
    print(f"{i}: {sentence}\n---")

#check number of sentences and approx tokens
num_sentences = len(sentences)
num_tokens = sum(len(sentence.split()) for sentence in sentences)

print(f"Number of sentences (archive): {num_sentences}")
print(f"Number of space-split tokens (archive): {num_tokens}")