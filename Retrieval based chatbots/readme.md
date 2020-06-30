Here I will create a Retrieval based chat bot with the help of Deep Neural network algorithm.
(For clearer insights follow the pyhton notebook)

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

# Building and training the model

Now we have the training data ready. Now a deep neural network, that has 3 layers, is built. 
This particular network has 3 layers, with the first one having 128 neurons, the second one having 64 neurons, and the third one having the number of intents as the number of neurons. Remember, the point of this network is to be able to predict which intent to choose given some data.
The model will be trained with stochastic gradient descent and categorical_crossentropy loss function. Stochastic gradient descent is more efficient than normal gradient descent.
Keras sequential API is used. After training the model for 200 epochs, 100% accuracy is  achieved on the model. The final epoch obtained the Loss function of 0.0132.
The trained model is saved as ‘chatbot_model.h5’ to skip the rigourous training of 2 hrs.

# Creating GUI for interacting with our chatbot
Now a graphical user interface is created. For this,the Tkinter library is used which already comes in python. It will take the input message from the user and then use the model created will predict the answers and display it on the GUI. (the same appproach is followed here.)
Anaconda or Google colab will not display the GUI properly. So, better to use Spyder that comes with Anaconda. Don't forget to launch it in Tensorflow environment)

The trained model is loaded and then a graphical user is made. The interface will predict the response from the bot. 
The model will only tell us the class it belongs to, so some functions are implemented which will identify the class and then retrieve us a random response from the list of responses.
After predicting the class, a random response is generated from the list of intents.
Here are some functions that contain all of the necessary processes for running the GUI and encapsulates them into units. We have the clean_up_sentence() function which cleans up any sentences that are inputted. This function is used in the bow() function, which takes the sentences that are cleaned up and creates a bag of words that are used for predicting classes (which are based off the results we got from training our model earlier).
In our predict_class() function, we use an error threshold of 0.25 to avoid too much overfitting. This function will output a list of intents and the probabilities, their likelihood of matching the correct intent. The function getResponse() takes the list outputted and checks the json file and outputs the most response with the highest probability.
Finally our chatbot_response() takes in a message (which will be inputted through our chatbot GUI), predicts the class with our predict_class() function, puts the output list into getResponse(), then outputs the response. What we get is the foundation of our chatbot. We can now tell the bot something, and it will then respond back.

# Folder structure
Here’s a quick breakdown of the components:
train_chatbot.py — the code for reading in the natural language data into a training set and using a Keras sequential neural network to create a model

chatgui.py — the code for cleaning up the responses based on the predictions from the model and creating a graphical interface for interacting with the chatbot

classes.pkl — a list of different types of classes of responses

words.pkl — a list of different words that could be used for pattern recognition

intents.json — abunch of JavaScript objects that lists different tags that correspond to different types of word patterns

chatbot_model.h5 — the actual model created by train_chatbot.py and used by chatgui.py

# Deployment or integration with any application or website
To deploy it in any website or application we can easily do it using TensorFlow lite. The trained model is saved in the main folder of Generative Based chat bot. It can be also deployed using any API service in which the model needs to be hosted in a server and connect it via a API to get the response. The GUI needs to be configured while deployment. Feel free to leave a comment in case of deployment issues.
