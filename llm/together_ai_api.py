import openai

# Use Together API key (not OpenAI)
openai.api_key = "58d8c30682973a7de3fa88615a006deb69a6453d4b8ee95843b6dc311cbedd66"
openai.api_base = "https://api.together.xyz/v1"

# Choose your model (e.g. LLaMA-2-70B or LLaMA-3-70B-Instruct)
model = "togethercomputer/llama-2-70b-chat"  # or "meta-llama/Llama-3-70b-instruct"

response = openai.ChatCompletion.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the capital of Iceland?"}
    ],
    temperature=0.7,
    max_tokens=100,
)

print(response["choices"][0]["message"]["content"])
