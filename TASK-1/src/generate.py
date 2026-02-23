from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def generate_text(prompt, max_length=50):
    model_path = "outputs/final_model"

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    prompt = input("Enter prompt: ")
    print("\nGenerated:\n")
    print(generate_text(prompt))