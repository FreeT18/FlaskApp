#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install nltk


# In[ ]:


pip install -U pip setuptools wheel


# In[ ]:


pip install pyqtwebengine==5.12.1


# In[ ]:


#In command prompt
#python -m spacy download en_core_web_sm


# In[ ]:


#import re
#cmdline = re.sub(r'[/"\'`]', ' ', cmdline).lower()
#words= nltk.tokenize.word_tokenize(cmdline)


# In[ ]:


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random


# In[ ]:


import os
os.getcwd()


# In[ ]:


os.chdir("./Downloads")


# In[ ]:


words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)


# In[ ]:


import re
import tensorflow as tf
import random


# In[ ]:


with open(r'Intents.json') as f:
    intents = json.load(f)


# In[ ]:


for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# In[ ]:


print(documents)


# In[ ]:


classes


# In[ ]:


# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))


# In[ ]:


print(words)


# In[ ]:


# sort classes
classes = sorted(list(set(classes)))


# In[ ]:


# documents = combination between patterns and intents
print (len(documents), "documents")


# In[ ]:


# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary


# In[ ]:


#Save into pickle files
print (len(words), "unique lemmatized words", words)
pickle.dump(words,open('texts.pkl','wb'))
pickle.dump(classes,open('labels.pkl','wb'))
# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)


# In[ ]:


# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    #Append everything to the training list
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)


# In[ ]:


# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


# In[ ]:


model = Sequential()
#Create input size based on training data
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


# In[ ]:


# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[ ]:


#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=120, batch_size=5, verbose=1)
model.save('model.h5', hist)
print("model created")


# Model Complete Here
# Next section for initializing and running the bot

# In[ ]:


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


# In[ ]:


words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))
model = load_model('model.h5')


# In[ ]:


def clean_up(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


# In[ ]:


def bag_of_words(sentence):
    sentence_words = clean_up(sentence)
    bag = [0] * len(words)
    for word in sentence_words:
        if word in words:
            index = words.index(word)
            bag[index] = 1
    return np.array(bag)


# In[ ]:


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


# In[ ]:


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            break
    return response


# In[ ]:


def log_chat(user_input, bot_response):
    log_file = open("chat_logs.txt", "a")
    log_file.write("User: " + user_input + "\n")
    log_file.write("Bot: " + bot_response + "\n\n")
    log_file.close()


# In[ ]:


while True:
    message = input("Enter your message(enter 'q' to quit): ")
    if message == 'q':
        break
    intents_list = predict_class(message)
    response = get_response(intents_list, intents)
    log_chat(message, response)
    print("Response:", response)


# Following section for database connection

# In[ ]:


import sqlite3

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
    store_log(message, response)


