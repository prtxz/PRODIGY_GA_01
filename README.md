# GPT-2 Fine-Tuning on Custom Gen-Z Dataset

## Overview

This project demonstrates fine-tuning of OpenAI's GPT-2 model on a custom Gen-Z conversational dataset. The objective was to train the model to generate coherent and stylistically consistent text based on informal Gen-Z language patterns.

The model was fine-tuned using the Hugging Face Transformers library and trained locally on CPU.

---

## Objective

Train a pre-trained GPT-2 model to generate contextually relevant text in a specific conversational style using a custom dataset.

---

## Dataset

- Custom curated Gen-Z style text dataset
- Informal conversational tone
- Short, reaction-based and expressive sentences
- Each line represents one training sample
- Approximately 250 samples used for fast training configuration

Example dataset entries:
bro that’s lowkey insane
nah this actually ate
bestie be so serious right now
this is giving main character energy


---

## Methodology

### 1. Tokenization

- Used GPT-2 tokenizer (Byte Pair Encoding)
- Maximum sequence length: 64 tokens
- Padding token set to EOS token
- Labels set equal to `input_ids` for next-token prediction

### 2. Training Configuration

- Base model: GPT-2 (124M parameters)
- Epochs: 1
- Batch size: 4
- Learning rate: 5e-5
- Optimizer: AdamW (default in Hugging Face Trainer)
- Training performed on CPU
- Save strategy disabled for faster execution

### 3. Fine-Tuning Objective

The model was trained using causal language modeling (next-token prediction).  
For each sequence, the model learns to predict the next token based on previous context.

---

## Project Structure
TASK-1/
│
├── data/
│ └── raw/
│ └── genz_dataset.txt
│
├── outputs/
│ └── final_model/
│
├── src/
│ ├── init.py
│ ├── dataset.py
│ ├── train.py
│ └── generate.py
│
├── README.md
└── requirements.txt

---

## How to Run

### 1. Install Dependencies

### 2. Train the Model

### 3. Generate Text

Then enter a prompt when prompted.

---

## Sample Outputs

**Prompt:**bro this is

**Output:**
bro this is not about me

**Prompt:**
you really thought


**Output:**

you really thought this was gonna be my second season


---

## Conclusion

The fine-tuned GPT-2 model successfully adapted to the stylistic patterns of the custom Gen-Z dataset. Despite training on a relatively small dataset and running on CPU, the model demonstrated consistent tone adaptation and context-aware text generation.

This project showcases the complete fine-tuning workflow:

- Dataset preparation  
- Tokenization  
- Model fine-tuning  
- Inference and text generation  
