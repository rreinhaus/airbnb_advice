from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from airbnb_advice.trainer import lines

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

# Loading the deep learning model
model = load_model('models_testdeep_model_best(1).h5')

# Input from the user for the prediction - THIS NEEDS TO BE CHANGED FROM THE STREAMLIT INPUT
room_types = ['Entire Place', 'Private Room', 'Shared Room' ]

def generate_text_seq(model, tokenizer, text_seq_length, seed_text, n_words):
  text = []

  for _ in range(n_words):
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    encoded = pad_sequences([encoded], maxlen=text_seq_length, truncating='pre')

    y_predict = np.argmax(model.predict(encoded), axis=-1)

    predicted_word = ''
    for word, index in tokenizer.word_index.items():
      if index == y_predict:
        predicted_word = word 
        break
    seed_text = seed_text + ' ' + predicted_word
    text.append(predicted_word)

  return ' '.join(text)


if __name__ == '__main__':
    result = generate_text_seq(model, tokenizer, 6, seed_text=room_types[0], n_words=7)
    title = room_types[0] + ' - ' + result
    print(title)