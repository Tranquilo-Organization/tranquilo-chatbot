from flask import Flask, render_template, request, jsonify
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from keras.models import load_model
import pickle
import logging

# Initialize the Flask application
app = Flask(__name__)
app.static_folder = "static"

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the lemmatizer and load resources
lemmatizer = WordNetLemmatizer()
model = load_model("model.h5")  # Load the trained model

# Load intents and model data
with open("intents.json") as f:
    intents = json.load(f)

with open("texts.pkl", "rb") as f:
    words = pickle.load(f)

with open("labels.pkl", "rb") as f:
    classes = pickle.load(f)


# Clean and tokenize the user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# Create a bag of words array: 0 or 1 for each word in the sentence
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    logger.debug("Found in bag: %s", w)
    return np.array(bag)


# Predict the intent of the user input
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# Get a response based on the intent
def get_response(ints, intents_json):
    if ints:
        tag = ints[0]["intent"]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                return result
    return "Sorry, I didn't understand that."


# Generate chatbot response based on the input message
def chatbot_response(msg):
    try:
        response = get_response(predict_class(msg, model), intents)
        logger.debug("Chatbot response: %s", response)
        return response
    except Exception as e:
        logger.error("Error generating response: %s", str(e))
        return "An error occurred while processing your request."


# Route for the homepage
@app.route("/")
def home():
    return render_template("index.html")


# Route to get the bot response for a user message via POST request with raw JSON
@app.route("/get", methods=["POST"])
def get_bot_response_post():
    try:
        data = request.get_json()  # Parse the JSON from the request body
        if not data or "msg" not in data:
            return (
                jsonify({"response": "Invalid input. Please provide a 'msg' field."}),
                400,
            )

        user_text = data["msg"]  # Get the 'msg' field from the JSON
        logger.debug("User input (POST, JSON): %s", user_text)

        chatbot_response_text = chatbot_response(user_text)
        return jsonify({"response": chatbot_response_text})

    except Exception as e:
        logger.error("Exception occurred: %s", str(e))
        return (
            jsonify({"response": "An error occurred while processing your request."}),
            500,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
