Templates

The chat pipeline guide introduced TextGenerationPipeline and the concept of a chat prompt or chat template for conversing with a model. Underlying this high-level pipeline is the apply_chat_template method. A chat template is a part of the tokenizer and it specifies how to convert conversations into a single tokenizable string in the expected model format.

In the example below, Mistral-7B-Instruct and Zephyr-7B are finetuned from the same base model but they’re trained with different chat formats. Without chat templates, you have to manually write formatting code for each model and even minor errors can hurt performance. Chat templates offer a universal way to format chat inputs to any model.

Copied
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

tokenizer.apply_chat_template(chat, tokenize=False)
Copied
<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]
This guide explores apply_chat_template and chat templates in more detail.

apply_chat_template

Chats should be structured as a list of dictionaries with role and content keys. The role key specifies the speaker (usually between you and the system), and the content key contains your message. For the system, the content is a high-level description of how the model should behave and respond when you’re chatting with it.

Pass your messages to apply_chat_template to tokenize and format them. You can set add_generation_prompt to True to indicate the start of a message.

Copied
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta", device_map="auto", torch_dtype=torch.bfloat16)

messages = [
    {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate",},
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(tokenized_chat[0]))
Copied
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s>
<|user|>
How many helicopters can a human eat in one sitting?</s>
<|assistant|>
Now pass the tokenized chat to generate() to generate a response.

Copied
outputs = model.generate(tokenized_chat, max_new_tokens=128) 
print(tokenizer.decode(outputs[0]))
Copied
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s>
<|user|>
How many helicopters can a human eat in one sitting?</s>
<|assistant|>
Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all.
add_generation_prompt

The add_generation_prompt parameter adds tokens that indicate the start of a response. This ensures the chat model generates a system response instead of continuing a users message.

Not all models require generation prompts, and some models, like Llama, don’t have any special tokens before the system response. In this case, add_generation_prompt has no effect.

Copied
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
tokenized_chat
Copied
<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
continue_final_message

The continue_final_message parameter controls whether the final message in the chat should be continued or not instead of starting a new one. It removes end of sequence tokens so that the model continues generation from the final message.

This is useful for “prefilling” a model response. In the example below, the model generates text that continues the JSON string rather than starting a new message. It can be very useful for improving the accuracy for instruction following when you know how to start its replies.

Copied
chat = [
    {"role": "user", "content": "Can you format the answer in JSON?"},
    {"role": "assistant", "content": '{"name": "'},
]

formatted_chat = tokenizer.apply_chat_template(chat, tokenize=True, return_dict=True, continue_final_message=True)
model.generate(**formatted_chat)
You shouldn’t use add_generation_prompt and continue_final_message together. The former adds tokens that start a new message, while the latter removes end of sequence tokens. Using them together returns an error.
TextGenerationPipeline sets add_generation_prompt to True by default to start a new message. However, if the final message in the chat has the “assistant” role, it assumes the message is a prefill and switches to continue_final_message=True. This is because most models don’t support multiple consecutive assistant messages. To override this behavior, explicitly pass the continue_final_message to the pipeline.