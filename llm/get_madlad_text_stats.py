import json
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
json_path = "llm/other_outputs/madlad_text_stats_20250630_154134.json"


# --- LOAD JSON ---
with open(json_path, "r", encoding="utf-8") as f:
    stats = json.load(f)


# --- 1. ZIPF'S LAW PLOT ---
zipf_data = stats["zipf"]
ranks = zipf_data["ranks"]
freqs_observed = zipf_data["frequencies_observed"]
freqs_expected = zipf_data["frequencies_zipf_expected"]

plt.figure(figsize=(8, 6))
plt.plot(ranks, freqs_observed, marker='o', linestyle='none', markersize=2, alpha=0.6, label="Observed")
plt.plot(ranks, freqs_expected, color='red', linestyle='--', linewidth=1.5, label="Zipf's Law (1/r)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Rank (log)")
plt.ylabel("Frequency (log)")
plt.title(f"Zipf's Law: Observed vs. Expected (Top {zipf_data['top_words_analyzed']})")
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# --- 2. CHARACTER DISTRIBUTION PLOT ---
char_dist = stats["character_distribution"]
chars = sorted(char_dist.keys())
proportions = [char_dist[ch] for ch in chars]

plt.figure(figsize=(12, 6))
plt.bar(chars, proportions, color='mediumseagreen')
plt.title("Proportional Character Frequency (a–z)")
plt.xlabel("Character")
plt.ylabel("Proportion")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# --- 3. SUMMARY STATISTICS PRINT ---
print("\n--- OVERALL TEXT STATISTICS ---")
overall = stats["overall"]
for k, v in overall.items():
    print(f"{k.replace('_', ' ').capitalize()}: {v:,.2f}" if isinstance(v, (int, float)) else f"{k}: {v}")

print("\n--- LANGUAGE DETECTION ---")
lang = stats["language_detection"]
print("Sentence counts:", lang["sentence_counts"])
print("Sentence proportions:")
for k, v in lang["sentence_proportions"].items():
    print(f"  {k}: {v:.2%}")

print(f"\nZipf distance: {zipf_data['zipf_distance']:.6f}")
