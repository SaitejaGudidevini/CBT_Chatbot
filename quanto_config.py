
import torch
print(torch.backends.mps.is_available())  # Should return True

from transformers import AutoModelForCausalLM
model_path = "/Users/MacBook/dev/TejaFiles/finetuning_llama3/llama3.2-3B-embed_tokens" 
model = AutoModelForCausalLM.from_pretrained(model_path)


from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
model_path = model_path
tokenizer = AutoTokenizer.from_pretrained(model_path) # Load tokenizer as well
quantization_config = QuantoConfig(weights="int8")
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="mps" # Explicitly move to MPS
)



save_directory = "./llama3.2-3B-embed_tokens-quantoconfig" 
quantized_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory) # Save the tokenizer as well


loaded_quantized_model.eval()
compiled_model = torch.compile(loaded_quantized_model, mode="reduce-overhead")