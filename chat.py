import random
import numpy as np
import pickle
import json
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

model = load_model("data/model.h5")
intents = json.loads(open("data/intents.json").read())
words = pickle.load(open("data/texts.pkl", "rb"))
classes = pickle.load(open("data/labels.pkl", "rb"))

with open('data/intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data/data.pth"

intents = None
tags = None
model = None
all_words = None

bot_name = "Ada"

def clean_up(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for w in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    result = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(result) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


""" def initialize(data_path, intents_path):
    global intents, tags, model, all_words

    with open(intents_path, 'r') as json_data:
        intents = json.load(json_data)

    data = torch.load(data_path)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"] """

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    #initialize()
    while True:
        message = input("You: ")
        if message == "quit":
            break
        ints = predict_class(message)
        result = get_response(ints, intents)
        print(result)
