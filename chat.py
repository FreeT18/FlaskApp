import numpy as np
import random
import json
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('data/texts.pkl','rb'))
classes = pickle.load(open('data/labels.pkl','rb'))
model = load_model('data/model.h5')

def clean_up(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up(sentence)
    bag = [0] * len(words)
    for word in sentence_words:
        if word in words:
            index = words.index(word)
            bag[index] = 1
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
    for intent in list_of_intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            break
    return response

def log_chat(user_input, bot_response):
    log_file = open("chat_logs.txt", "a")
    log_file.write("User: " + user_input + "\n")
    log_file.write("Bot: " + bot_response + "\n\n")
    log_file.close()


while True:
    message = input("Enter your message(enter 'q' to quit): ")
    if message == 'q':
        break
    intents_list = predict_class(message)
    response = get_response(intents_list, intents)
    log_chat(message, response)
    print("Response:", response)


# Following section for database connection
""" import sqlite3

def store_log(message, response):
    conn = sqlite3.connect('chatlogs.db')
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS logs(message text, response text)")
    c.execute("INSERT INTO logs(message, response) VALUES (?,?)", (message, response))
    conn.commit()
    conn.close()

while True:
    message = input("Enter your message(enter 'q' to quit): ")
    if message == 'q':
        break
    intents_list = predict_class(message)
    response = get_response(intents_list, intents)
    print("Response:", response)
    store_log(message, response) """