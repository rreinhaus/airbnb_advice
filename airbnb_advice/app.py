# from tkinter.tix import INTEGER
import streamlit as st
import datetime
import pandas as pd
import requests
from google.cloud import storage

from shapely.geometry import Point, Polygon
import geopandas as gpd
import pandas as pd
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import requests
import json
import pydeck as pdk
import numpy as np
from airbnb_advice.trainer import lines
from pred import generate_text_seq 

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

st.markdown('''
# AIR BNB ADVIC€
## Welcolme to our amazing app
Richard, Nicolas, Joana and Thomas

''')

st.markdown('''
#Thanks to provide the data in the inbox below so Artificial Intelligence can predict the TAXI FARE  : 
''')
# st.write(df.head())

###############################################
#####  collection of the data form the USER 
country=st.text_input('select a country','UK')
city_user = st.selectbox('select a city',  ["London",""] )
address = st.text_input("adress", "Fill in the adress of your housing")
full_adress = address + city_user + country
#####fonction pour récupére l'API
if st.button('best keywords for the city'):
    url = "https://airbnbadvice-zktracgm3q-ew.a.run.app/keywords/?city="+city_user
    response = requests.get(url).json()
    city_keywords = response["keywords"]
    text_to_show = 'the best keywords for '+city_user+' found by our artifical inteligence are : '
    st.text(text_to_show) #show the text of the  API
    st.text(city_keywords)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

# Loading the deep learning model
model = load_model('/home/thomas/code/thomasgassin/airbnb_advice/airbnb_advice/models_testdeep_model_best(1).h5')
adds = st.text_input("adds", "Fill in two keywords")
if st.button('the best announce will be '):
    st.text(generate_text_seq(model, tokenizer, 6, seed_text=adds, n_words=7)) 
# Getting pick up location as address and transforming to coordinates

loc = Nominatim(user_agent= "GetLoc" )
geocode = RateLimiter(loc.geocode, min_delay_seconds=1)
location = loc.geocode(address +","+city_user+","+ country )

latitude = location.latitude
longitude = location.longitude

st.markdown(latitude)
st.markdown(longitude)

entire_home = st.checkbox("Chek if you rent the entire home", value = False ) # binary does the user rent the full accomodation or not
nb_bedrooms = st.slider("number or rooms", 1,10,2) #the number of room
nb_beds = st.slider("how many beds",1,10,2) # the number of beds
min_nights = st.slider("minimum night", 1,7,1)
accomodates = int(st.number_input('how many guests can you accomodate' , min_value=0, value=5, step=1 ))

#########################################
# if st.button('Artifial Intelligence will compute best fare for your accomodation'):
#     url = f"""https://airbnbadvice-zktracgm3q-ew.a.run.app/
#             fare_prediction/?latitude=%{latitude}&
#             longitude={longitude}&
#             accomodates={accomodates}&
#             bedrooms={nb_bedrooms}&
#             beds={nb_beds}&
#             minimum_nights={min_nights}&
#             Entire_home_apt={min_nights}"""
#     response = requests.get(url).json()
#     fare_predicted = response['predicted_fare']
#     st.text("the predicted price should be ")
#     st.text("fare_predicted") 


if st.button('Artifial Intelligence will compute best fare for your accomodation'):
    url = f"https://airbnbadvice-zktracgm3q-ew.a.run.app/fare_prediction/?latitude={latitude}&longitude={longitude}&accomodates={accomodates}&bedrooms={nb_bedrooms}&beds={nb_beds}&minimum_nights={min_nights}&Entire_home_apt={min_nights}"
    st.text(url)
    response = requests.get(url).json()
    fare_predicted = response['predicted_fare']
    st.text("the predicted price should be ")
    st.text(fare_predicted) 













json_api_request  = {  "latitude" : latitude ,
                            "longitude" : longitude ,
                            "accomodates": accomodates,
                            "nb_bedrooms" : nb_bedrooms , 
                            "nb_beds" : nb_beds,
                            "minimum_nights" : min_nights , 
                            "Entire_home_apt" : entire_home
                            }


# reviews = int(st.number_input('Please insert the number of reviews of your housing' , min_value=0, value=5, step=1 ))
# amenities_string = st.text_input('Amenities :', 'Enter the amenities available at your housing')
# st.write('Amenities available are', amenities_string)
# rent_starting_date = st.date_input(
#     "When do you want to rent",
#     datetime.date(2022, 2 , 18))
# st.write('The starting date is ', rent_starting_date)
# max_stay = st.slider("maximum stay" , 1,21,7)

