import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader

# Load the dataset
df = pd.read_csv("traffic.csv")

# Assuming your CSV has columns named 'Question' and 'Answer'
question_answer_pairs = list(zip(df["Question"], df["Answer"]))

# Define the custom dataset class
class QADataset(Dataset):
    def __init__(self, qa_pairs, tokenizer, max_length=128):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]
        # Format the input text as "Question: {question} Answer: {answer}"
        input_text = f"Question: {question} Answer: {answer}"
        # Tokenize the input text
        encodings = self.tokenizer(input_text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        input_ids = encodings.input_ids.squeeze(0)  # Remove batch dimension
        attention_mask = encodings.attention_mask.squeeze(0)  # Remove batch dimension
        return {"input_ids": input_ids, "attention_mask": attention_mask}

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Prepare the data loader
dataset = QADataset(question_answer_pairs, tokenizer)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

# Fine-tuning the model
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 25 # You can adjust the number of epochs based on your dataset size and desired performance
for epoch in range(num_epochs):
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_gpt2_qa")
