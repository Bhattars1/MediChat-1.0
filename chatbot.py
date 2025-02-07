from transformers import BioGptForCausalLM, AutoTokenizer
import torch

model_path = "model/best_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BioGptForCausalLM.from_pretrained(model_path)

import re

def generate_text(input_text, model= model, tokenizer=tokenizer):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        output = model.generate(input_ids,
                                max_length=1024,
                                temperature=0.7,
                                top_k=50,
                                top_p=0.9,
                                do_sample=True)
        response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # Remove the input text if it appears at the beginning
    if response.startswith(input_text):
        response = response[len(input_text):].strip()

    # Use regex to extract Context, Answer, and Decision
    match = re.search(
        r"Context:\s*(.*?)\s*(Answer:\s*.*?)?\s*(Decision:\s*yes|Decision:\s*no)",
        response, re.DOTALL
    )

    if match:
        context = match.group(1).strip()
        answer = match.group(2).strip()
        decision = match.group(3).strip()

        # Return formatted response
        formatted_response = f"{context}\n\n{answer}\n\n{decision}"
        return formatted_response
    else:
        return "Could not format the response correctly."

print("MediChat: Type 'exit', 'quit', or 'bye' to stop.")

while True:
    user_input = input("User: ")
    
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Bot: Goodbye!")
        break

    bot_response = generate_text(user_input)
    print(f"Bot: {bot_response}")
