from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

# Input from the user for the prediction - THIS NEEDS TO BE CHANGED FROM THE STREAMLIT INPUT
room_types = ['Entire Place', 'Private Room', 'Shared Room' ]

def generate_text_seq(input_text):
  lines = pd.read_csv('airbnb_advice/data/lines.csv')
  lines = lines['lines'].values.tolist()

  # Tokenizing the input on the lines and achieving the sequences required for the model
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(lines)
  sequences = tokenizer.texts_to_sequences(lines)

  # Loading the deep learning model
  model = load_model('models_testdeep_model_best(1).h5')

  text = []

  for _ in range(8):
    encoded = tokenizer.texts_to_sequences([input_text])[0]
    encoded = pad_sequences([encoded], maxlen=6, truncating='pre')

    y_predict = np.argmax(model.predict(encoded), axis=-1)

    predicted_word = ''
    for word, index in tokenizer.word_index.items():
      if index == y_predict:
        predicted_word = word 
        break
    input_text = input_text + ' ' + predicted_word
    text.append(predicted_word)

  return ' '.join(text)


if __name__ == '__main__':
    result = generate_text_seq(room_types[0])
    title = room_types[0] + ' - ' + result
    print(title)