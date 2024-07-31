import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# cd .venv cd scripts conda activate aienv

# Load preprocessed data (tokenized_text)
with open('text.txt', 'r') as file:
    lines = file.readlines()

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Encode tokenized text
max_length = 256  # Set an appropriate value
input_ids = [tokenizer.encode(sentence, add_special_tokens=True, padding='max_length', max_length=max_length) for sentence in lines]

# Filter out empty sentences
input_ids = [ids for ids in input_ids if len(ids) > 0]

# Convert to tensors
input_tensors = torch.tensor(input_ids)

# Check if input data is empty
if len(input_tensors) == 0:
    print("No valid input data. Please check your preprocessed text.")
else:
    # Training loop (customize as needed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 5
    batch_size = 4

    for epoch in range(num_epochs):
        for batch_start in range(0, len(input_tensors), batch_size):
            batch = input_tensors[batch_start : batch_start + batch_size]
            optimizer.zero_grad()
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Batch {batch_start//batch_size+1}: Loss = {loss.item():.4f}")

    # Evaluate the model (calculate perplexity)
    with torch.no_grad():
        loss = model(input_ids=input_tensors, labels=input_tensors).loss
        perplexity = torch.exp(loss)
        print(f"Perplexity on test data: {perplexity:.2f}")

    # Save the model
    model.save_pretrained("gpt2_pretrained_model")


# Now you can interact with the saved model
# Example:
# generator = pipeline('text-generation', model='gpt2_pretrained_model')
# generated_text = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
# print(generated_text)
