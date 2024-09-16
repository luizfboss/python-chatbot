import random # choose random response
import json # open JSON file
import pickle
import numpy as np

import nltk
# Run this block of code ONCE to prevent errors and bugs with nltk in the future
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer # consider words the same regardless of grammar

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer() # NLTK Constructor

intents = json.loads(open('intents.json').read()) # Reading data that will be processed when training network

words = [] # all words we will have
classes = [] # all classes we will have
documents = [] # combinations/belongings
ignore_letters = ['?', '!', '.', ','] # letters/chars to be ignored

# organizing data that will be used to train AI
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) # this will get a text and it will split it into collection of words/individual words
        words.extend(word_list) # adding words to the words_list
        documents.append((word_list, intent['tag'])) # appending collection of words to the documents too, but now assigned to the 'tag'

        # check if class is already in class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters] # removing letters we don't want to take into consideration (chars from ignore_letters)
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb')) # saving words as pickle files and writing as binary
pickle.dump(classes, open('classes.pkl', 'wb')) # saving classes as pickle files and writing as binary

# Machine learning part - we need to transorm all information into numerical values in order to work on its own.
# Create a bag of words for this.
training = []
output_empty = [0] * len(classes)

# when we run a loop all the document data will be in the training array to train the neural network.
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(output_empty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

train_x = training[:, :len(words)]
train_y = training[:, len(words):]

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

model.save('chatbot_model.keras', hist) # message from the console - save file as .keras instead of .h5 -> this is a legacy format :)
print("Done")

