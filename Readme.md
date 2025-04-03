# Interactive CBT Therapy Conversation with Fine-Tuned LLaMA 3.2

This script provides an interactive interface to simulate a Cognitive Behavioral Therapy (CBT) conversation using a fine-tuned LLaMA 3.2 model. The model has been fine-tuned on CBT-style conversations and further optimized using quantization for efficient inference.

---

## Features

- **Interactive Conversation**: Engage in a real-time conversation with the fine-tuned model.
- **Dynamic System Prompt**: Update the system's behavior dynamically during the session.
- **Conversation History Management**: Clear the conversation history while retaining the system prompt.
- **Response Cleaning**: Automatically cleans and optimizes the model's responses to remove repetitive patterns and special tokens.
- **Quantization**: Uses `QuantoConfig` for 8-bit quantization, optimized for MPS (Metal Performance Shaders) on macOS.

---

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Required Python libraries:
  - `transformers`
  - `torch`
  - `peft`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/finetuning-llama3.git
   cd finetuning-llama3