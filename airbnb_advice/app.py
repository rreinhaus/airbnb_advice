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


st.markdown('''
# AIR BNB ADVIC€
## Welcolme to our amazing app
Richard, Nicolas, Christelle and Thomas

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

json_api_request  = {  "latitude" : latitude ,
                            "longitude" : longitude ,
                            "accomodates": accomodates,
                            "nb_bedrooms" : nb_bedrooms , 
                            "nb_beds" : nb_beds,
                            "min_nights" : min_nights , 
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

