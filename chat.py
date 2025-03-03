import random
import json
from transformers import pipeline

# Load intents
with open("intents.json", "r") as json_data:
    intents = json.load(json_data)

# Load a better lightweight LLM model (Mistral 7B Instruct)
generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

bot_name = "SlingBot"

# System prompt to guide the LLM
system_prompt = (
    "You are SlingBot, a helpful AI assistant for Sling TV. "
    "You provide clear and concise responses about Sling TV services. "
    "Avoid repetition and keep answers relevant."
)

def get_response(msg):
    # Check if the input matches any predefined intents
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            if pattern.lower() in msg.lower():
                return random.choice(intent["responses"])

    # Generate response using the model with a structured prompt
    full_prompt = f"{system_prompt}\n\nUser: {msg}\n{bot_name}:"
    response = generator(full_prompt, max_length=100, do_sample=True, temperature=0.7)

    # Extract and clean response
    generated_text = response[0]["generated_text"].replace(full_prompt, "").strip()
    return generated_text.split("\n")[0]  # Take only the first sentence to avoid repetition

if __name__ == "__main__":
    print("SlingBot is ready! (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print(f"{bot_name}: Goodbye! Have a great day!")
            break

        resp = get_response(user_input)
        print(f"{bot_name}: {resp}")
