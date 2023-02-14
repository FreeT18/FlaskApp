from flask import Flask, render_template, request, jsonify, url_for
#from chat import get_response, initialize
import numpy as np
import random
import json
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

app = Flask(__name__)


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('data/intents.json').read())

words = pickle.load(open('data/texts.pkl', 'rb'))
classes = pickle.load(open('data/labels.pkl', 'rb'))
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

@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    intents_list = predict_class(message)
    response = get_response(intents_list, intents)
    return response

""" @app.route('/chatbot', methods=['POST'])
def chatbot_response():
    user_input = request.form.get('text')
    response = chatbot.get_response(user_input)
    return str(response) """

if __name__ == "__main__":
    app.run(debug=True)
