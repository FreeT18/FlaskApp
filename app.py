from flask import Flask, render_template, request, jsonify, url_for

from chat import get_response, initialize

app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")


@app.post("/predict")
def predict():
    INTENTS_PATH = url_for('static', filename='data/intents.json')
    DATA_PATH = url_for('static', filename='data/data.pth')
    initialize(DATA_PATH, INTENTS_PATH)
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

""" @app.route('/chatbot', methods=['POST'])
def chatbot_response():
    user_input = request.form.get('text')
    response = chatbot.get_response(user_input)
    return str(response) """


if __name__ == "__main__":
    app.run(debug=True)
