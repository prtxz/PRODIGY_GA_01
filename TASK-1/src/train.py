from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
from src.dataset import prepare_dataset


def train_model():
    dataset, tokenizer = prepare_dataset("data/raw/genz_dataset.txt")

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    training_args = TrainingArguments(
        output_dir="outputs",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        logging_steps=20,
        learning_rate=5e-5,
        weight_decay=0.01,
        prediction_loss_only=True,
        save_strategy="no",
        use_cpu=True  # Set to False if you have a GPU available
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
    )

    trainer.train()

    model.save_pretrained("outputs/final_model")
    tokenizer.save_pretrained("outputs/final_model")


if __name__ == "__main__":
    train_model()