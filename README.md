# Fine-Tuning LLaMA 3.2 for CBT Therapy Conversations

This project demonstrates the fine-tuning of the LLaMA 3.2 model for Cognitive Behavioral Therapy (CBT) conversations. The goal is to create a conversational AI that can simulate a CBT therapist, guiding patients through structured conversations to help them manage their thoughts and emotions.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Fine-Tuning Process](#fine-tuning-process)
- [Inference and Interaction](#inference-and-interaction)
- [Quantization](#quantization)
- [File Descriptions](#file-descriptions)
- [Future Improvements](#future-improvements)

---

## Project Overview

This project fine-tunes the LLaMA 3.2 model on a dataset of CBT-style conversations. The fine-tuned model is capable of:
- Asking focused questions to understand the patient's situation.
- Providing brief, step-by-step guidance.
- Simulating a CBT therapist's conversational style.

The project also includes quantization to optimize the model for inference on devices with limited resources, such as MPS (Metal Performance Shaders) on macOS.

---

## Features

- **Fine-Tuned LLaMA 3.2**: A conversational AI fine-tuned for CBT therapy.
- **Quantization**: Reduces model size and improves inference speed using `QuantoConfig`.
- **Interactive Conversation**: A script for real-time interaction with the fine-tuned model.
- **Customizable System Prompts**: Allows users to modify the system's behavior dynamically.

---

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- Required Python libraries:
  - `transformers`
  - `torch`
  - `datasets`
  - `peft`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/finetuning-llama3.git
   cd finetuning-llama3


pip install -r requirements.txt

import torch
print(torch.backends.mps.is_available())  # Should return True

#### Fine-Tuning Process
Dataset Preparation:

The dataset (cbt_finetuning_dataset.json) contains structured CBT conversations.
Each conversation includes roles (system, Patient, CBT Therapist) and their respective messages.
Fine-Tuning Script:

The script finetuning_sfttrainer.py fine-tunes the LLaMA 3.2 model using the dataset.
Key steps:
Load the dataset and preprocess it.
Tokenize the data using the LLaMA tokenizer.
Train the model using the transformers library.
Output:

The fine-tuned model is saved in the directory llama3.2-3B-embed_tokens.
Inference and Interaction
Quantization:

The script quanto_config.py applies QuantoConfig to the fine-tuned model for 8-bit quantization.
The quantized model is saved in llama3.2-3B-embed_tokens-quantoconfig.
##### Interactive Conversation:

The script Inference_training.py allows real-time interaction with the quantized model.
Features:
-Dynamic system prompt updates.
-Conversation history management (clear/reset).
-Response cleaning to remove repetitive patterns and special tokens.
-Run the Interactive Script:

python Inference_training.py

##### Quantization
Quantization reduces the model size and improves inference speed. This project uses QuantoConfig for 8-bit quantization. The quantized model is optimized for MPS (Metal Performance Shaders) on macOS.

File Descriptions
finetuning_sfttrainer.py:

Fine-tunes the LLaMA 3.2 model on the CBT dataset.
cbt_finetuning_dataset.json:

The dataset containing CBT-style conversations.
quanto_config.py:

Applies quantization to the fine-tuned model.
Inference_training.py:

Provides an interactive interface for real-time conversations with the quantized model.
llama3.2-3B-embed_tokens:

Directory containing the fine-tuned model.
llama3.2-3B-embed_tokens-quantoconfig:

Directory containing the quantized model.

