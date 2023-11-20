import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

# Variables to keep track of whether 'hi' and 'help' tags have been used
hi_tag_used = False
help_tag_used = False

def get_response(msg):
    global hi_tag_used, help_tag_used  # Declare the variables as global

    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        # Check if 'hi' tag has been used
        if tag.lower() == "hi" or tag == "Hi":
            hi_tag_used = True  # Set the flag to True
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])

        # Check if 'help' tag has been used
        elif tag.lower() == "help":
            help_tag_used = True  # Set the flag to True
            for intent in intents['intents']:
                if intent["tag"] == "help":
                    return random.choice(intent['responses'])

    # Allow access to OptionA, OptionB, OptionC only for 'help' tag
    if help_tag_used and tag in ["Option A", "Option B", "Option C"]:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    # Allow access to all intents after 'hi' tag has been used and 'help' tag has been used
    elif hi_tag_used or help_tag_used:
        for intent in intents['intents']:
            if tag == intent["tag"] and tag not in ["Option A", "Option B", "Option C"]:
                return random.choice(intent['responses'])

    return "Sorry🤔 I couldn't understand, please rephrase your question."

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
