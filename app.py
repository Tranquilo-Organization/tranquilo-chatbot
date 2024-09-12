from flask import Flask, render_template, request, jsonify
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from keras.models import load_model
import pickle
import logging
import os

# Initialize the Flask application
app = Flask(__name__)
app.static_folder = "static"

# Setup logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logs
logger = logging.getLogger(__name__)

# Initialize resources
lemmatizer = WordNetLemmatizer()

# Load trained model and resources
try:
    model = load_model("model.h5")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error("Error loading the model: %s", str(e))
    raise RuntimeError(f"Failed to load model: {str(e)}")

try:
    with open("intents.json") as f:
        intents = json.load(f)
    logger.info("Intents file loaded successfully.")
except Exception as e:
    logger.error("Error loading intents.json: %s", str(e))
    raise RuntimeError(f"Failed to load intents file: {str(e)}")

try:
    with open("texts.pkl", "rb") as f:
        words = pickle.load(f)
    with open("labels.pkl", "rb") as f:
        classes = pickle.load(f)
    logger.info("Model support files (texts.pkl, labels.pkl) loaded successfully.")
except Exception as e:
    logger.error("Error loading model support files: %s", str(e))
    raise RuntimeError(f"Failed to load support files: {str(e)}")


# Helper functions
def clean_up_sentence(sentence):
    """Tokenize and lemmatize the user input."""
    try:
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        logger.debug("Cleaned sentence: %s", sentence_words)
        return sentence_words
    except Exception as e:
        logger.error("Error cleaning sentence: %s", str(e))
        raise


def bow(sentence, words, show_details=True):
    """Create a bag of words from the input sentence."""
    try:
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        logger.debug("Found in bag: %s", w)
        logger.debug("Bag of words: %s", bag)
        return np.array(bag)
    except Exception as e:
        logger.error("Error creating bag of words: %s", str(e))
        raise


def predict_class(sentence, model):
    """Predict the intent of the input sentence."""
    try:
        p = bow(sentence, words, show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        logger.info("Prediction results: %s", results)
        return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    except Exception as e:
        logger.error("Error predicting class: %s", str(e))
        raise


def get_response(ints, intents_json):
    """Generate a response based on the predicted intent."""
    try:
        if ints:
            tag = ints[0]["intent"]
            for intent in intents_json["intents"]:
                if intent["tag"] == tag:
                    response = random.choice(intent["responses"])
                    logger.info("Generated response: %s", response)
                    return response
        logger.warning("No matching intent found.")
        return "Sorry, I didn't understand that."
    except Exception as e:
        logger.error("Error generating response: %s", str(e))
        raise


def chatbot_response(msg):
    """Generate a response from the chatbot."""
    try:
        ints = predict_class(msg, model)
        response = get_response(ints, intents)
        logger.info("Chatbot response: %s", response)
        return response
    except Exception as e:
        logger.error("Error generating chatbot response: %s", str(e))
        return "An error occurred while processing your request."


# Routes
@app.route("/")
def home():
    """Serve the homepage."""
    try:
        logger.info("Rendering home page.")
        return render_template("index.html")
    except Exception as e:
        logger.error("Error rendering home page: %s", str(e))
        return "An error occurred while rendering the homepage.", 500


@app.route("/chatbot", methods=["POST"])
def get_bot_response_post():
    """Handle chatbot responses via POST with both form and JSON support."""
    try:
        if request.is_json:  # If the request contains JSON data
            data = request.get_json()
            user_text = data.get("msg", "")
        else:  # If it's from an HTML form (application/x-www-form-urlencoded)
            user_text = request.form.get("msg", "")

        if not user_text:
            return (
                jsonify({"response": "Invalid input. Please provide a 'msg' field."}),
                400,
            )

        logger.debug("User input (POST): %s", user_text)
        chatbot_response_text = chatbot_response(user_text)
        return jsonify({"response": chatbot_response_text})

    except Exception as e:
        logger.error("Error processing request: %s", str(e))
        return (
            jsonify({"response": "An error occurred while processing your request."}),
            500,
        )


# Application entry point
if __name__ == "__main__":
    try:
        logger.info("Starting Flask application.")
        # Use environment variable for port and disable debug mode for production
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
    except Exception as e:
        logger.critical("Failed to start Flask application: %s", str(e))
        raise
