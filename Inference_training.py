from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig

# Define model names directly instead of importing from finetuning_sfttrainer
model_name = "meta-llama/Llama-3.2-3B"
tokenizer_name = model_name + "-Instruct"  # We use Instruct tokenizer for its chat template

quantized_model_path = "./llama3.2-3B-embed_tokens-quantoconfig"                                                                        
adapter_path = "/Users/MacBook/dev/TejaFiles/finetuning_llama3/llama3.2-3B-embed_tokens/checkpoint-26"

# Load the tokenizer directly
tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


                                                                                                                                        
# For MPS, QuantoConfig is the best option available                                                                                    
quantization_config = QuantoConfig(weights="int8") 


# Load the base Model
base_model = AutoModelForCausalLM.from_pretrained(
    quantized_model_path,
    device_map='mps',
    #torch_dtype=torch.float32
)
print(f"Model is in training mode:{base_model.training}")
base_model.eval()
print(f"Model is in training mode after .eval(): {base_model.training}")                                                                

# Load the adapter
model = PeftModel.from_pretrained(base_model, adapter_path)



def clean_response(response):
    """Clean the response by removing special tokens and repetitive patterns"""
    # Find the assistant's response
    assistant_start = response.find("<|start_header_id|>assistant<|end_header_id|>")
    if assistant_start == -1:
        return response
    
    # Extract just the assistant's message
    assistant_content = response[assistant_start + len("<|start_header_id|>assistant<|end_header_id|>"):]
    
    # Remove any special tokens at the end
    import re
    clean_content = re.sub(r'<\|reserved_special_token_\d+\|>.*$', '', assistant_content)
    
    # Remove repetitive patterns (like "What evidence do you have" repeated)
    clean_content = re.sub(r'(\b\w+(?:\s+\w+){3,6}\b)(?:\s*\1\b)+', r'\1', clean_content)
    
    # If the response is still too long, truncate it
    words = clean_content.split()
    if len(words) > 50:
        clean_content = ' '.join(words[:50])
    
    return clean_content.strip()

def interactive_conversation():
    """Run an interactive CBT therapy conversation"""
    print("\n===== CBT Therapy Conversation =====")
    print("Type your thoughts or concerns. Type 'exit' to end the session.")
    print("Type 'system' to change the system prompt.")
    print("Type 'clear' to clear the conversation history.\n")
    
    # Default system message
    default_system_message = "You are a CBT therapist. Ask one focused question at a time to understand the patient's situation. Keep your responses brief (2-3 sentences maximum). Wait for the patient to respond before asking another question. Focus on guiding the conversation step by step rather than providing lengthy explanations."
    
    # Initialize conversation with system message
    conversation = [
        {
            "role": "system",
            "content": default_system_message
        }
    ]
    
    while True:
        # Get user input
        user_input = input("Patient: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nSession ended. Take care!")
            break
        elif user_input.lower() == "system":
            print("\nEnter new system prompt:")
            new_system_prompt = input("> ")
            # Update the system message
            conversation = [
                {
                    "role": "system",
                    "content": new_system_prompt
                }
            ]
            print("\nSystem prompt updated and conversation history cleared.")
            continue
        elif user_input.lower() == "clear":
            # Reset conversation but keep the current system prompt
            system_prompt = conversation[0]["content"]
            conversation = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]
            print("\nConversation history cleared.")
            continue
        
        # Add user input to conversation history
        conversation.append({"role": "Patient", "content": user_input})
        
        # Prepare messages for the model
        messages = conversation.copy()
        
        # Apply chat template
        input_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate response
        inputs = tokenizer(input_chat, return_tensors="pt").to("mps")
        #memorty optimization for MPS
        import gc
        gc.collect()

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                pad_token_id=tokenizer.eos_token_id, 
                max_new_tokens=150,  # Reduced from 600
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,  # Add repetition penalty
                no_repeat_ngram_size=3,  # Prevent repeating the same 3-gram
                early_stopping=False      # Stop when the model would naturally end the response
            )
        
        # Decode and clean response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        therapist_response = clean_response(full_response)
        
        # Print the response
        print("\nCBT Therapist:")
        print(therapist_response)
        print()
        
        # Add therapist response to conversation history
        conversation.append({"role": "CBT Therapist", "content": therapist_response})

        gc.collect() 

# Run the interactive conversation
if __name__ == "__main__":
    interactive_conversation()
