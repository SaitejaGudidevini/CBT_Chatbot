"""
Fine-tuning Llama 3.2 with Supervised Fine-Tuning (SFT)
This script demonstrates how to fine-tune Llama 3.2 using supervised fine-tuning (SFT) to create an education chatbot. We will cover:
1. Loading and formatting a question-answering dataset
2. Applying and appropriate chat template
3. Setting up LoRA fine-tuning with special token training
4. Training the model
5. Testing the fine-tuned model
"""

# Install required packages if not already installed
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Uncomment these lines if you need to install packages
# install_package("peft")
# install_package("trl")

# Load Dataset
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from datasets import Dataset
import pandas as pd

# Define model and tokenizer names
model_name = "meta-llama/Llama-3.2-3B"
tokenizer_name = model_name + "-Instruct"  # We use Instruct tokenizer for its chat template

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the JSON file
json_path = os.path.join(script_dir, 'cbt_finetuning_dataset_littleconvo.json')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

#Setting padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Pad token set to: {tokenizer.pad_token}")




# Open the JSON file
with open(json_path, 'r') as f:
    conversations = json.load(f)

# Check if the file is empty or the content is not as expected
"""
if not conversations:
    print("The JSON file is empty or its content could not be loaded correctly.")
else:
    print(f"Loaded {len(conversations)} conversations. Preview of the first conversation:")
    print(json.dumps(conversations[0], indent=2))
"""

# Process the conversations into a format suitable for training
processed_data = []

# Function to trim long responses
def trim_long_responses(response, max_words=50):
    """Limit responses to a maximum number of words"""
    words = response.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return response

# Process each conversation
for conversation in conversations:
    # Extract system message (first message with role "system")
    system_msg = next((msg["content"] for msg in conversation if msg["role"] == "system"), "")
    
    # Process the conversation into pairs of Patient-Therapist exchanges
    for i in range(len(conversation) - 1):
        if conversation[i]["role"] == "Patient" and conversation[i+1]["role"] == "CBT Therapist":
            processed_data.append({
                "system": system_msg,
                "question": conversation[i]["content"],
                "answer": trim_long_responses(conversation[i+1]["content"])
            })

# Import train_test_split
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42)

# Convert to Huggingface Datasets
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))

print(f"Training examples: {len(train_dataset)}")
print(f"Testing examples: {len(test_dataset)}")


#Define the format_with_chat_template function
def format_with_chat_template(example):
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example['question']},
        {"role": "assistant", "content": example['answer']}
    ]

    #Apply chat template without tokenizing
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"formatted_text": formatted_text}

# Apply formatting to both datasets
formatted_train_dataset = train_dataset.map(format_with_chat_template)
formatted_test_dataset = test_dataset.map(format_with_chat_template)

# Tokenize both datasets
def tokenize_function(examples):
    return tokenizer(examples['formatted_text'], truncation=True, max_length=2048)

tokenized_train_dataset = formatted_train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = formatted_test_dataset.map(tokenize_function, batched=True)

print(f"Tokenized training dataset: {tokenized_train_dataset}")
print(f"Tokenized testing dataset: {tokenized_test_dataset}")

# Update system prompt for better conversation structure
system_prompt = "You are a CBT therapist. Ask one focused question at a time to understand the patient's situation. Keep your responses brief (2-3 sentences maximum). Wait for the patient to respond before asking another question."


##configure LoRA finetuning
"""We will setup LoRA fine-tuning with special tokenn training by unfreezing the token embeddings"""
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
#configure LoRa
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
    #modules_to_save=['embed_tokens']  # Unfreeze token embeddings to train special tokens
)

#Load Model for Training
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='mps',
    torch_dtype=torch.float32

)
#Prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

model.print_trainable_parameters()


#Setting up Training Arguments
from transformers import TrainingArguments

output_dir = "./llama3.2-3B-embed_tokens"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,  # Add eval batch size
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=100,
    logging_steps=10,
    evaluation_strategy="steps",  # Add evaluation strategy
    eval_steps=100,               # Evaluate every 100 steps
    fp16=False,
    bf16=False,
    optim='adamw_torch',
    lr_scheduler_type='linear',
    warmup_ratio=0.1,
    report_to="tensorboard",
    push_to_hub=False,
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="eval_loss",  # Use eval loss to determine the best model
)

from transformers import DataCollatorForLanguageModeling                                                            
                                                                                                                    
data_collator = DataCollatorForLanguageModeling(                                                                    
    tokenizer=tokenizer,                                                                                            
    mlm=False  # Not using masked language modeling                                                                 
)   

from transformers import Trainer                                                                                    
                                                                                                                    
trainer = Trainer(                                                                                                  
    model=model,                                                                                                    
    args=training_args,                                                                                             
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,  # Add the test dataset for evaluation                                                                                
    data_collator=data_collator,                                                                                    
)     

trainer.train()      

                                                                                                                    
model.save_pretrained(output_dir)                                                                                   
tokenizer.save_pretrained(output_dir)   

