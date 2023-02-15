from flask import Flask, render_template, request, jsonify, url_for
#from chat import get_response, initialize
import numpy as np
import random
import json
import pickle

from flask_ngrok import run_with_ngrok
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

#functions
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('data/intents.json').read())
words = pickle.load(open('data/texts.pkl', 'rb'))
classes = pickle.load(open('data/labels.pkl', 'rb'))
model = load_model('data/model.h5')

app = Flask(__name__)
#run_with_ngrok(app)

def clean_up(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words, show_details=True):
    sentence_words = clean_up(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    bow = bag_of_words(sentence, words, show_details=False)
    result = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(result) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            break
    return response

def log_chat(message, response):
    log_file = open("chat_logs.txt", "a")
    log_file.write("User: " + message + "\n")
    log_file.write("Bot: " + response + "\n\n")
    log_file.close()

#define routes
@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    if msg.startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = get_response(ints, intents)
        res =res1.replace("{n}",name)
    elif msg.startswith('hi my name is'):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = get_response(ints, intents)
        res =res1.replace("{n}",name)
    else:
        ints = predict_class(msg, model)
        res = get_response(ints, intents)
    return res


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    intents_list = predict_class(message)
    response = get_response(intents_list, intents)
    log_chat(message, response)
    return response

""" @app.route('/chatbot', methods=['POST'])
def chatbot():
    message = request.form['message']
    response = get_chatbot_response(message)
    return response """

if __name__ == "__main__":
    app.run(debug=True)
