#0. INITIAL SET-UP --------------------------------------------------------------

#import libraries
import json
from collections import Counter
import string
import unicodedata
import matplotlib.pyplot as plt
import gzip
import numpy as np
import math
import re
from fast_langdetect import detect

#set filepaths
madlad_data_path_zip_clean = "llm/data/madlad_from_huggingface/gd_clean_0000.jsonl.gz"

#load data
with gzip.open(madlad_data_path_zip_clean, "rt", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

#flatten data
data = " ".join(item.get("text", "") for item in data)


#1. OVERALL STATS --------------------------------------------------------------

#get overall stats
total_words = len(data.split())
total_chars = total_chars = len(data)

#average word length
words = data.split()
total_word_chars = sum(len(word) for word in words)
total_words = len(words)
average_word_length = total_word_chars / total_words if total_words > 0 else 0

#average sentence length
sentences = re.split(r'[.!?]+[\s\n]+', data)
sentences = [s.strip() for s in sentences if s.strip()]
total_sentences = len(sentences)
words_per_sentence = [len(s.split()) for s in sentences]
average_sentence_length = sum(words_per_sentence) / total_sentences if total_sentences > 0 else 0


#2. CHARACTER DISTRIBUTION ------------------------------------------------------

#get character distribution
data_lower = data.lower()
char_counts = Counter(ch for ch in data_lower if ch in string.ascii_lowercase)
total_letter_count = sum(char_counts.values())
char_proportions = {ch: char_counts[ch] / total_letter_count for ch in string.ascii_lowercase}

#plot character distribution
chars = list(string.ascii_lowercase)
proportions = [char_proportions.get(ch, 0) for ch in chars]
plt.figure(figsize=(12, 6))
plt.bar(chars, proportions, color='mediumseagreen')
plt.title("Proportional Character Frequency (a–z)")
plt.xlabel("Character")
plt.ylabel("Proportion")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#3. ZIPF'S LAW ------------------------------------------------------

#get unigram frequencies (zipf's law)
words = data.lower().split()
word_counts = Counter(words)
sorted_word_freqs = word_counts.most_common()
cutoff = 10000
sorted_word_freqs = sorted_word_freqs[:cutoff]
frequencies = [freq for _, freq in sorted_word_freqs]
ranks = np.arange(1, len(frequencies) + 1)

#plot unigram frequencies
C = frequencies[0]
zipf_freqs = [C / r for r in ranks]
plt.figure(figsize=(8, 6))
plt.plot(ranks, frequencies, marker='o', linestyle='none', markersize=2, alpha=0.6, label="Observed")
plt.plot(ranks, zipf_freqs, color='red', linestyle='--', linewidth=1.5, label="Zipf's Law (1/r)")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Rank (log)")
plt.ylabel("Frequency (log)")
plt.title(f"Word Frequency Distribution vs Zipf's Law (Top {cutoff})")
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#distance between observed and expected frequencies
observed = np.array(frequencies)
expected = np.array(zipf_freqs)
observed_norm = observed / observed.sum()
expected_norm = expected / expected.sum()
zipf_distance = np.linalg.norm(observed_norm - expected_norm)


#4. LANGUAGE DETECTION ------------------------------------------------------

#proportion of English (using sentence-wise lang detect)
counts = {"gd": 0, "en": 0, "other": 0}
for sentence in sentences:
    try:
        lang = fast_langdetect.detect(sentence)
    except Exception:
        lang = "other"
    
    if lang == "gd":       # Scottish Gaelic language code
        counts["gd"] += 1
    elif lang == "en":
        counts["en"] += 1
    else:
        counts["other"] += 1

total = sum(counts.values())
proportions = {k: v / total for k, v in counts.items()}
