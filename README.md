# IMDB-Movie-Review-using-Sentimental-Analysis

Classifying movie reviews as positive or negative using the text of the review using binary —classification  -Done using Deep Learning neural network using the IMDB Movie reviews using sentimental analysis.

In this project, I implemented 3 layer neural network using Keras.

Requirements
Jupyter Notebook (Python)
Keras
Numpy

Open : imdb-rating-sentiment-analysis (1) on Jupyter notebook


Installation instructions for Anaconda 3 Jupyter Notebook
We must use the following installation steps:
1.	Download Anaconda 3 from google chorme.
2.	Installation done for Jupyter Notebook, open it. 
Dataset
The "Large Movie Review Dataset" is used in this project. The data set contains the text of 50,000 movie reviews from the Internet Movie Database(IMDB) . These are split into 25,000 reviews for training and 25,000 reviews for testing Each review is preprocessed to be encoded as a sequence of word indexes. Each word is mapped into an integer that stands for how frequently the word is used. For instance, let there be a sentence "To be or not to be". The mapping of the words are as follows:

The dataset already exists among keras datasets. It was imported for use and the data cleaning from sentence to words to numbers was done before itself. Only text classification is needed.

Getting Started
Functions and libraries required to create the model were imported.

Loading IMDB Dataset
For loading the data we use command
print(imdb.load_data())
I kept the top 5000 words used from the dataset and ignored the rest for higher accuracy,and each review with a max length of 100 words also splited the dataset into two equal parts which are training and test sets.

Hyperparameters - 
max_features = 5000
maxlen = 100
batch_size = 64
embedding_dims = 16
filters = 128
kernel_size = 3
hidden_size = 128
epochs =10

Word Embedding
Since movie reviews are actually sequences of words, there is need to encode them. Word embedding has been used to represent features of the words with semantic vectors and map each movie review into a real vector domain. I specified embedding vector length as 16.

 So the first layer of the model is embedding layer. This will use 16 length vector to represent each word.

Building the model
The model has 3 layer.
 I used relu for the activation on the hidden layer. Details are in the code file. 

Creating LSTM Model
LSTM, which is an often used natural language processing technique for both sentiment analysis, text classification is used. LSTM is a special kind of recurrent neural network which is capable of learning long term dependencies. LSTM is able to remember information for long periods of time as a default behavior.

Activation function in dense layer is sigmoid. Adam, which is an adaptive learning method, was used as optimizer. Batch size was specified as 64.

I used optimizer as ‘adam'for compiling the model,
loss function as ='binary_crossentropy' as we are dealing  with binary classification.
metrics as 'accuracy'

Training the model
I trained the model using Mini-batch gradient descent for only 10 epochs with batch size of 64. Here's result of the training for the five epochs. 

We got accuracy of 100% for training data at the 10th epoch

Testing the model
I tested the model on the remaining 25,000 movie reviews.. and got testing Accuracy: 83.71%
