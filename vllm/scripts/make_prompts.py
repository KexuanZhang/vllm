from datasets import load_dataset

# Load dataset
ds = load_dataset("openai/gsm8k", "main", split="test")

# Select first 25 questions
prompts = [ex["question"] for ex in ds.select(range(25))]

# Write to file
output_file = "prompts.txt"
with open(output_file, "w") as f:
    f.write("\n\n---\n\n".join(prompts))

print(f"Saved {len(prompts)} prompts to {output_file}")
