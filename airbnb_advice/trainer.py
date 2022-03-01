import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import string
from google.cloud import storage


BUCKET_NAME = 'test-rreinhaus'

BUCKET_TRAIN_DATA_PATH = 'airbnb_advice/raw_data/listings.csv'

MODEL_NAME = 'title_generator'

MODEL_VERSION = 'v1'

def get_data():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}")
    title_data = df[['id', 'name']][df['review_scores_rating'] > 4]
    return title_data

def preprocess(df):
    """method that pre-process the data"""
    def corpus(text):
        """function generates corpus for the preprocessing"""
        text = ''.join(e for e in text if e not in string.punctuation).lower()
        text = text.encode('utf8').decode('ascii', 'ignore')
        return text

    corpus = [corpus(str(e)) for e in df['name']]

    data = " ".join(corpus)

    def clean_text(text):
        """function cleans and creates tokens"""
        tokens = text.split()
        table = str.maketrans("", "", string.punctuation)
        tokens = [w.translate(table) for w in tokens]
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [word.lower() for word in tokens]
        return tokens

    tokens = clean_text(data)
    length = 6+1
    lines = []

    for i in range(length, len(tokens)):
        seq = tokens[i-length:i]
        line = " ".join(seq)
        lines.append(line)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)
    sequences = np.array(sequences)

    X, y = sequences[:, :-1], sequences[:,-1]

    len_ = int(0.9*len(X))

    X_train = X[:len_]
    X_test = X[len_:]

    y_train = y[:len_]
    y_test = y[len_:]

    vocab_size = len(tokenizer.word_index) + 1

    y_train = to_categorical(y_train, num_classes=vocab_size)
    y_test = to_categorical(y_test, num_classes=vocab_size)
    
    return X_train, y_train, X_test, y_test, vocab_size, lines

def train_model(X_train, y_train, epochs, vocab_size):
    """method that trains the model"""

    seq_length = X_train.shape[1]
    model = Sequential()

    # Embedding Layer
    model.add(Embedding(vocab_size, 10, input_length=seq_length))
    
    # 2x LSTM models and with 20% dropout 
    model.add(LSTM(128,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))

    model.add(Dense(32,activation='relu'))
    model.add(Dense(vocab_size,activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(patience=3)

    model.fit(X_train, y_train, batch_size=128, epochs=epochs,callbacks=[es])

    return model

STORAGE_LOCATION = 'models/testdeep/model_best.h5'

def upload_model_to_gcp():


    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename('model_best.h5')

def save_model(model):
    """method that saves the model into a .h5 file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    
    model.save('model_best.h5')
    print("saved model.h5 locally")

    # Implement here
    upload_model_to_gcp()
    print(f"uploaded model_best.h5 to gcp cloud storage under \n => {STORAGE_LOCATION}")

df = pd.read_csv('/home/rreinhaus/code/rreinhaus/airbnb_advice/raw_data/listings.csv')
df = df[['id', 'name']][df['review_scores_rating'] > 4]

X_train, y_train, X_test,y_test, vocab_size, lines = preprocess(df)

if __name__ == '__main__':
    # get training data from GCP bucket
    df = pd.read_csv('/home/rreinhaus/code/rreinhaus/airbnb_advice/raw_data/listings.csv')
    df = df[['id', 'name']][df['review_scores_rating'] > 4]

    # preprocess data
    X_train, y_train, X_test,y_test, vocab_size, lines = preprocess(df)

    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)
    model = train_model(X_train, y_train, 1, vocab_size)

    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model(model)
