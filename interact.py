from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the pretrained model from the saved directory
model_path = "gpt2_pretrained_model"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained(model_path)

# Set pad_token_id
model.config.pad_token_id = 50256

# Now you can use the loaded model for text generation or other tasks
input_text = input()
input_ids = tokenizer.encode(input_text, return_tensors="pt")
attention_mask = torch.ones_like(input_ids)  # Create an attention mask
output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)

