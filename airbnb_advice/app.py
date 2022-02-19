from tkinter.tix import INTEGER
import streamlit as st
import datetime
# from shapely.geometry import Point, Polygon
# import geopandas as gpd
# import pandas as pd
# import geopy
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter
# import requests
# import json
# import pydeck as pdk


st.markdown('''
# AIR BNB ADVIC€
## Welcolme to our amazing app
Richard, Nicolas, Christelle and Thomas

''')

st.markdown('''
#Thanks to provide the data in the inbox below so Artificial Intelligence can predict the TAXI FARE  : 
''')
###############################################
#####  collection of the data form the USER 
city_user = st.selectbox('select a city',  ["London"] )
st.write("you are the owner of a housing in " , city_user)
address = st.text_input("adress", "Fill in the adress of your housing")
nb_bedrooms = st.slider("number or rooms", 1,10,2)

reviews = int(st.number_input('Please insert the number of reviews of your housing' , min_value=0, value=5, step=1 ))

amenities_string = st.text_input('Amenities :', 'Enter the amenities available at your housing')
amenities_list = amenities_string.split()
st.write('Amenities available are', amenities_string)



rent_starting_date = st.date_input(
    "When do you want to rent",
    datetime.date(2022, 2 , 18))
st.write('The starting date is ', rent_starting_date)

min_stay = st.slider("minimum stay", 1,7,1)

max_stay = st.slider("maximum stay" , 1,21,7)

########################################

json_for_api_request  = {   "city_user" : city_user ,
                            "adress" : address ,
                            "nb_bedrooms" : nb_bedrooms , 
                            "reviews" : reviews,
                            "amenities" : amenities_list ,
                            "rent_starting_date" : rent_starting_date ,
                            "min_stay" : min_stay ,
                            "max_stay" : max_stay
                            }



# # Defining the number of columns
# columns = st.columns(2)

# # Getting pick up location as address and transforming to coordinates

# column_pick = st.columns(2)

# street_pick = column_pick[0].text_input("Which street you want to travel from?", value="20 W 34th Street")
# city_pick = "New York"
# country_pick = "USA"

# geolocator = Nominatim(user_agent="GTA Lookup")
# geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
# location_pick = geolocator.geocode(street_pick+", "+city_pick+","+country_pick)

# lat_pick = location_pick.latitude
# lon_pick = location_pick.longitude