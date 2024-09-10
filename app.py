from flask import Flask, render_template, request, jsonify
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from keras.models import load_model
import pickle

# Initialize the lemmatizer and load resources
lemmatizer = WordNetLemmatizer()
model = load_model("model.h5")  # Load the trained model

# Load intents and model data
intents = json.loads(open("intents.json").read())
words = pickle.load(open("texts.pkl", "rb"))
classes = pickle.load(open("labels.pkl", "rb"))


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
                    print("found in bag: %s" % w)
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
def getResponse(ints, intents_json):
    if ints:
        tag = ints[0]["intent"]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
        return result
    else:
        return "Sorry, I didn't understand that."


# Generate chatbot response based on the input message
def chatbot_response(msg):
    res = getResponse(predict_class(msg, model), intents)
    print("Chatbot response: ", res)
    return res


# Flask Web Application
app = Flask(__name__)
app.static_folder = "static"


# Route for the homepage
@app.route("/")
def home():
    return render_template("index.html")


# Route to get the bot response for a user message via POST request with raw JSON
@app.route("/get", methods=["POST"])
def get_bot_response_post():
    data = request.get_json()  # Parse the JSON from the request body
    userText = data.get("msg")  # Get the 'msg' field from the JSON
    print("User input (POST, JSON): " + userText)

    chatbot_response_text = chatbot_response(userText)

    return jsonify({"response": chatbot_response_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
