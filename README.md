

```markdown
# 🧠 Tranquilo Chatbot

## 📖 Overview

Tranquilo is a chatbot designed to assist users by providing conversational responses based on user input. The chatbot leverages a pre-trained deep learning model and is built using Python, Flask for the backend, and deployed on Railway for public access. The model uses natural language processing (NLP) techniques to understand and respond to user queries based on defined intents.

## ✨ Features

- 🤖 AI-driven conversational chatbot
- 📚 Pre-trained deep learning model using TensorFlow and Keras
- 🚀 Deployed on Railway for easy access
- 🌐 Integrated with Flask for handling HTTP requests
- 📝 JSON-based intents to handle varied responses
- 🎨 HTML/CSS/JS frontend for user interaction

## 🔧 Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **AI/ML**: TensorFlow, Keras (with pre-trained chatbot model)
- **Deployment**: Railway (PaaS)
- **Data Storage**: JSON file for intents and labels

## 🧠 How the AI Works

1. **Model Training**: 
   - The chatbot uses a deep learning model based on NLP principles. The training data consists of various intents stored in the `intents.json` file.
   - Each intent has predefined patterns and responses. These patterns are processed and vectorized, allowing the model to predict which intent the user’s message matches.

2. **Model Files**: 
   - `model.h5`: This is the saved trained model file using Keras.
   - `chatbot_model.h5`: Another version of the trained chatbot model.
   - `labels.pkl` & `texts.pkl`: Serialized files used during prediction for matching user inputs with corresponding responses.

3. **Response Mechanism**: 
   - The chatbot uses the trained model to classify the user's input, and based on the intent, it provides appropriate responses.
   - Multiple responses are defined per intent, and the chatbot randomly selects one of them to increase variation.

## 📂 Directory Structure

```bash
.
├── templates              # Contains HTML files for the web interface
│   ├── static
│   │   ├── chatbot.js     # JavaScript logic for frontend
│   │   ├── index.html     # Frontend HTML file for interaction
├── app.py                 # Main Flask app and routing logic
├── chatbot_model.h5       # Pre-trained chatbot model file
├── Dockerfile             # (Not used in the current deployment phase)
├── intents.json           # Intents with patterns and responses
├── labels.pkl             # Encoded labels for training data
├── model.h5               # Another trained model file
├── Procfile               # For Railway deployment configuration
├── requirements.txt       # Python dependencies for the project
└── training.py            # Script for training the model
```

## 🚀 Setup and Installation

### 1. Clone the repository:

```bash
git clone https://github.com/your-repo/tranquilo-chatbot.git
cd tranquilo-chatbot
```

### 2. Install dependencies:

You need Python 3.7+ to run this project. Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### 3. Training the Model (Optional):

If you want to retrain the model:

```bash
python training.py
```

The trained model will be saved as `model.h5` or `chatbot_model.h5`.

### 4. Run the Flask App:

```bash
python app.py
```

Visit `http://127.0.0.1:5000/` in your browser to interact with the chatbot.

## 🌍 Deployment

This project is currently deployed on [Railway](https://railway.app/) using the following steps:

1. Create a `Procfile` in your project root directory to specify how to run the app:
    ```bash
    web: python app.py
    ```

2. Upload the project to Railway.

3. **Note**: The Docker file is ignored for this phase of deployment, and we use the `Procfile` to start the Flask server.

4. After deployment, your chatbot will be accessible online.

## 🔗 Endpoints

- **GET** `/`: Loads the chatbot interface.
- **POST** `/chatbot`: Takes user input as JSON and returns a chatbot response.

### Example POST Request

```bash
POST /chatbot
Content-Type: application/json

{
  "msg": "Hello"
}
```

### Example Response

```json
{
  "response": "Hi there, how can I assist you?"
}
```

## 🚧 Future Enhancements

- Integration with Docker for better endpoint and interaction handling.
- More diverse intent categories.
- Enhancements to the AI model for better understanding of complex queries.

## 🔔 Tip

**Note**: The current version may not work correctly in Postman or other API testing tools due to real-time interaction limitations. We plan to use Docker in the next competition phase to fix this and ensure better API functionality.
```

This README now contains the tip at the end and includes Markdown icons for enhanced readability and design!
