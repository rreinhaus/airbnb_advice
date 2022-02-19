import streamlit as st
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

st.slider("number or rooms", 1,10,2)

st.markdown(""" <style> .font {
font-size:50px ; font-family: 'Cooper Black'; color: #FF9633;} 
</style> """, unsafe_allow_html=True)

st.markdown('<p class="font">Guess the object Names</p>', unsafe_allow_html=True)







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