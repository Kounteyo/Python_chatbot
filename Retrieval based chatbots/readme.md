Here I will create a Retrieval based chat bot with the help of Deep Neural network algorithm.

# Preprocessing the data
The data file is in JSON format so I used the json package to parse the JSON file into Python.

1. Tokenization:
While working with text data, various preprocessing needs to be performed on the data before a machine learning or a deep learning model is made. 
Tokenizing is the most basic and first thing you can do on text data. Tokenizing is the process of breaking the whole text into small parts like words.
Here I iterate through the patterns and tokenize the sentence using nltk.word_tokenize() function and append each word in the words list. 
A list of classes for the tags is also created.

2. Lemmatization:
Now each word is lemmatize to remove duplicate words from the list. 
Lemmatizing is the process of converting a word into its lemma form and then creating a pickle file to store the Python objects which will used while predicting.

3. Vectoriztion:
Computer doesn’t understand text so text needs to be converted into vectors i.e. numbers.

# Create training and testing data
Now, the training data will be created in which we will provide the input and the output. 
Our input will be the vectors and output will be the class our input pattern belongs to.

# Build the model

Now we have the training data ready. Now a deep neural network, that has 3 layers, is built. 
Keras sequential API is used. After training the model for 200 epochs, 100% accuracy is  achieved on the model. 
The trained model is saved as ‘chatbot_model.h5’ to skip the rigourous training of 2 hrs.

# Creating GUI for interacting with our chatbot
Now a graphical user interface is created. For this,the Tkinter library is used which already comes in python. It will take the input message from the user and then use the model created will predict the answers and display it on the GUI. (the same appproach is followed here.)
Anaconda or Google colab will not display the GUI properly. So, better to use Spyder that comes with Anaconda. Don't forget to launch it in Tensorflow environment)

The trained model is loaded and then a graphical user is made. The interface will predict the response from the bot. 
The model will only tell us the class it belongs to, so some functions are implemented which will identify the class and then retrieve us a random response from the list of responses.
After predicting the class, a random response is generated from the list of intents.

# Folder structure
Here’s a quick breakdown of the components:
train_chatbot.py — the code for reading in the natural language data into a training set and using a Keras sequential neural network to create a model

chatgui.py — the code for cleaning up the responses based on the predictions from the model and creating a graphical interface for interacting with the chatbot

classes.pkl — a list of different types of classes of responses

words.pkl — a list of different words that could be used for pattern recognition

intents.json — abunch of JavaScript objects that lists different tags that correspond to different types of word patterns

chatbot_model.h5 — the actual model created by train_chatbot.py and used by chatgui.py
