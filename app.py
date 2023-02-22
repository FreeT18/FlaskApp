from flask import Flask, render_template, request, jsonify, g
import sqlite3
import os
import numpy as np
import random
import json
import pickle
from datetime import datetime
#from flask_ngrok import run_with_ngrok
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

DATABASE = '/data/logs.db'

app = Flask(__name__)
#run_with_ngrok(app)

#Initialize the database
conn = sqlite3.connect('data/logs.db')

#Create the database model
""" class ChatLogs(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    text = db.Column(db.String(1000), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class Response(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    text = db.Column(db.String(1000), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
 """

#Define functions
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('data/intents.json').read())
words = pickle.load(open('data/texts.pkl', 'rb'))
classes = pickle.load(open('data/labels.pkl', 'rb'))
model = load_model('data/model.h5')

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
   with open('data/chat_logs.txt', 'a') as file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{timestamp} - User: {message}\n")
        file.write(f"{timestamp} - Bot: {response}\n\n")

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

#define routes
@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def chatlog():
    message = request.form['message']
    response = request.form['response']
    connection = sqlite3.connect(currentdirectory="\ChatLogs.db")
    cursor = connection.cursor()

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

@app.route('/save_logs', methods=['POST'])
def save_logs():
    data = request.get_json()
    with open('data/logs.txt', 'a') as f:
        f.write(data['log'] + '\n')
    return jsonify({'success': True})

""" @app.route('/chatbot', methods=['POST'])
def chatbot():
    message = request.form['message']
    response = get_chatbot_response(message)
    return response """


if __name__ == "__main__":
    app.run(debug=True)
