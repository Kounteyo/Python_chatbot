Here I will assemble a seq2seq LSTM model using Keras Functional API to create a working Chatbot which would answer questions asked to it.

Chatbots have become applications themselves. You can choose the field or stream and gather data regarding various questions. 
We can build a chatbot for an e-commerce webiste or a school website where parents could get information about the school.

# Preprocessing the data
The data can be downloaded from the chatterbot/english on Kaggle.com by kausr25. 
It contains pairs of questions and answers based on a number of subjects like food, history, AI etc.

The raw data could be found from this repo -> https://github.com/shubham0204/Dataset_Archives

After impoting the data we will read the yaml files and convert them into a list of questions and answers.

The seq2seq model requires three arrays namely encoder_input_data, decoder_input_data and decoder_output_data.

For encoder_input_data :
  Tokenize the questions. Pad them to their maximum length.

For decoder_input_data :
  Tokenize the answers. Pad them to their maximum length.

For decoder_output_data :
  Tokenize the answers. Remove the first element from all the tokenized_answers. This is the <START> element which we added earlier.
  
# Creating the Encoder-Decoder Model
The model will have Embedding, LSTM and Dense layers. The basic configuration is as follows.

2 Input Layers : One for encoder_input_data and another for decoder_input_data.
Embedding layer : For converting token vectors to fix sized dense vectors. ( Note : Don't forget the mask_zero=True argument here )
LSTM layer : Provide access to Long-Short Term cells.
Working :

The encoder_input_data comes in the Embedding layer ( encoder_embedding ).
The output of the Embedding layer goes to the LSTM cell which produces 2 state vectors ( h and c which are encoder_states )
These states are set in the LSTM cell of the decoder.
The decoder_input_data comes in through the Embedding layer.
The Embeddings goes in LSTM cell ( which had the states ) to produce seqeunces.

# Training the model
The model is trained for 150 number of epochs (it can be increased if required) with RMSprop optimizer and categorical_crossentropy loss function.
The final epoch obtained the Loss function of 0.0132

# Creating model to predict answers
An inference model is created which help in predicting answers.

Encoder inference model : Takes the question as input and outputs LSTM states ( h and c ).

Decoder inference model : Takes in 2 inputs, one are the LSTM states ( Output of encoder model ), second are the answer input seqeunces ( ones not having the <start> tag ). It will output the answers for the question which is fed to the encoder model and its state values.

# creating GUI for talking with our chatbot
Now a graphical user interface is created. For this,the Tkinter library is used which already comes in python. It will take the input message from the user and then use the model created will predict the answers and display it on the GUI. 
(the same appproach is followed here.)

Anaconda or Google colab will not display the GUI properly. 
So, better to use Spyder that comes with Anaconda. Don't forget to launch it in Tensorflow environment)

# Folder structure
Chatbot 2.py is the whole chatbot code as self executable python file. 
Chatbot 2.ipynb is for better illustration (originally created in Google colab)

The main model is saved as:
model.h5 (this for skipping rigourous 2 hours of training)

The encoder model is saved as: 
The main tensorflow model: enc_model.h5
TensorFlow lite version: enc_model.tflite

The decoder model is saved as: 
The main tensorflow model: dec_model.h5
TensorFlow lite version: dec_model.tflite

# Deployment or integration with any application or website

To deploy it in any website or application we can easily do it using TensorFlow lite. The trained model is saved in the main folder of Generative Based chat bot.
It can be also deployed using any API service in which the model needs to be hosted in a server and connect it via a API to get the response.
The GUI needs to be configured while deployment. 
Feel free to leave a comment in case of deployment issues.




