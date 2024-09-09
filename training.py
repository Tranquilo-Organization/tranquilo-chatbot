import numpy as np
import random
import json
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents file
intents = json.loads(open("intents.json").read())

# Lists to hold data
words = []
classes = []
documents = []
ignore_words = ["?", "!", ".", ","]

# Process intents data
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize each word
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add documents and associate with intent's tag
        documents.append((word_list, intent["tag"]))
        # Add to our classes if the tag is not already included
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize and lower each word, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Create training set, bag of words for each sentence
for doc in documents:
    # Initialize bag of words
    bag = []
    # List of tokenized words for the pattern
    word_patterns = doc[0]
    # Lemmatize each word and create the bag of words array
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for w in words:
        bag.append(1) if w in word_patterns else bag.append(0)

    # Output is a '0' for each tag and '1' for the current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)

# Convert training data into numpy arrays
training = np.array(training, dtype=object)

# Split into features (X) and labels (y)
train_x = np.array([item[0] for item in training], dtype=np.float32)
train_y = np.array([item[1] for item in training], dtype=np.float32)

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# Compile model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train the model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save("chatbot_model.h5")

print("Model created successfully")
