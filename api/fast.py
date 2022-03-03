from fastapi import FastAPI
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from  google.cloud import storage
import pandas as pd
from google.cloud import storage
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

# @app.get("/")
# def index():
#     return{"London":"keywords_of_Richard"}

url = "gs://airbnbadvice/data/description_london.csv"

def open_data_from_url():
    data = pd.read_csv(url)
    # resultats = data["keywords"].to_list()
    # print(resultats)
    return data

open_data_from_url()