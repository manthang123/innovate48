import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Recreate the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the fine-tuned model
model = GPT2LMHeadModel.from_pretrained("fine_tuned_gpt2_qa")

# Function to generate an answer given a question
def generate_answer(question, max_length=50):
    input_text = f"Question: {question} Answer:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate answer
    output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
 
    # Remove the question part from the answer
    answer = answer.split("Answer:")[1].strip()

    return answer

# Example usage
question = "What is the maximum permissible blood alcohol limit for a driver below 18 years of age?"
answer = generate_answer(question)
print("Answer:", answer)


from google.colab import drive
drive.mount('/content/drive')


import shutil

# Zip the fine-tuned model folder
shutil.make_archive("fine_tuned_gpt2_qa", 'zip', "fine_tuned_gpt2_qa")



import shutil

# Zip the fine-tuned model folder
shutil.make_archive("fine_tuned_gpt2_qa", 'zip', "fine_tuned_gpt2_qa")
