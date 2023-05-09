from flask import Flask, render_template, request, jsonify, g
import sqlite3
#import os
import openpyxl
import pandas as pd
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

DATABASE = './data/logs.db'

app = Flask(__name__)
#run_with_ngrok(app)

#Initialize the database
conn = sqlite3.connect(DATABASE, check_same_thread=False)

#Create the database model
try:
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE chatbot_log
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                   msg TEXT,
                   response TEXT,
                   timestamp DATETIME)''')
    conn.commit()
except:
    cursor = conn.cursor()

query = "SELECT * from chatbot_log where id >= 10"
cursor.execute(query)
rows = cursor.fetchall()

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
    bag = [1 if word in sentence_words else 0 for word in words]
    if show_details:
      found_words = [words[i] for i in range(len(words)) if bag[i] == 1]
      print("found in bag: {}".format(', '.join(found_words)))
    return np.array(bag)

def predict_class(sentence, model, ERROR_THRESHOLD=0.25):
    bow = bag_of_words(sentence, words, show_details=False)
    result = model.predict(np.array([bow]))[0]
    #ERROR_THRESHOLD = 0.25
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
        """elif intent['tag'] != tag:
            response = "Can you please clarify your question?"
            break"""
    return response

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

@app.route("/producth")
def producth():
    return render_template("producth.html")

@app.route("/productu")
def productu():
    return render_template("productu.html")

@app.route("/productx")
def productx():
    return render_template("productx.html")

@app.route("/pricing")
def pricing():
    return render_template("pricing.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    if msg.lower().startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        #res1 = get_response(ints, intents)
        response = "Hello " + name.title() + ". How can I help you?"
    elif msg.lower().startswith('hi my name is'):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = get_response(ints, intents)
        #response =res1.replace("{n}",name)
        response = "Hello " + name.title() + ". How can I help you?"
    else:
        ints = predict_class(msg, model)
        response = get_response(ints, intents)
    # Insert the user message and bot response to the chat log table
    timestamp = datetime.now()
    cursor.execute("INSERT INTO chatbot_log (msg, response, timestamp) VALUES (?, ?, ?)",
                   (msg, response, timestamp))
    conn.commit()
    try:
        chat_logs_df = pd.read_excel('chat_logs.xlsx')
    except:
        chat_logs_df = pd.DataFrame(columns=['Timestamp', 'User', 'Bot'])

    new_row = [[timestamp,msg,response]]
    new_row = pd.DataFrame(new_row, columns=['Timestamp','User', 'Bot'])
    frames=[chat_logs_df,new_row]
    chat_logs_df = pd.concat(frames)
    chat_logs_df.to_excel('chat_logs.xlsx', index=False)
    return response

@app.route("/get_rows")
def get_rows():
    """
    query = "SELECT * from chatbot_log"
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    return jsonify(rows)

if __name__ == "__main__":
    app.run(debug=True)
