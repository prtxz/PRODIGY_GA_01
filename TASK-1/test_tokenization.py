from src.dataset import prepare_dataset

dataset, tokenizer = prepare_dataset("data/raw/genz_dataset.txt")

print(dataset["train"][0].keys())
print(dataset["train"][0])