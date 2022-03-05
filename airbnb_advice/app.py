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
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static

st.set_page_config(
            page_title="Airbnb Advice", # => Quick reference - Streamlit
            page_icon="🐍",
            layout="centered", # wide
            initial_sidebar_state="auto") # collapsed




st.markdown('''
# AIR BNB ADVIC€
## Welcolme to our amazing app
Richard, Nicolas, Joana and Thomas

''')

st.sidebar.markdown('''
Thanks to provide the data in the inbox below so Artificial Intelligence can predict the TAXI FARE  :
''')
# st.write(df.head())

###############################################
#####  collection of the data form the USER in the SIDEBAR
country=st.sidebar.text_input('select a country','UK')
city_user = st.sidebar.selectbox('select a city',  ["London",""] )
address = st.sidebar.text_input("adress", "Fill in the adress of your housing")
full_adress = address + city_user + country

entire_home = st.sidebar.checkbox("Chek if you rent the entire home", value = False ) # binary does the user rent the full accomodation or not
nb_bedrooms = st.sidebar.slider("number or rooms", 1,10,2) #the number of room
nb_beds = st.sidebar.slider("how many beds",1,10,2) # the number of beds
min_nights = st.sidebar.slider("minimum night", 1,7,1)
accomodates = int(st.sidebar.number_input('how many guests can you accomodate' , min_value=0, value=5, step=1 ))



#####fonction pour récupére l'API
if st.button('best keywords for the city'):
    url = "https://airbnbadvice-zktracgm3q-ew.a.run.app/keywords/?city="+city_user
    response = requests.get(url).json()
    city_keywords = response["keywords"]
    text_to_show = 'the best keywords for '+city_user+' found by our artifical inteligence are : '
    st.text(text_to_show) #show the text of the  API
    st.text(city_keywords)

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(lines)
# sequences = tokenizer.texts_to_sequences(lines)

# Loading the deep learning model
# model = load_model('/home/thomas/code/thomasgassin/airbnb_advice/airbnb_advice/models_testdeep_model_best(1).h5')
# adds = st.text_input("adds", "Fill in two keywords")
# if st.button('the best announce will be '):
#     st.text(generate_text_seq(model, tokenizer, 6, seed_text=adds, n_words=7))
# Getting pick up location as address and transforming to coordinates
neighbourhood = None
loc = Nominatim(user_agent= "GetLoc" )
geocode = RateLimiter(loc.geocode, min_delay_seconds=1)
location = loc.geocode(address +","+city_user+","+ country,addressdetails=True)

if location is None:
    st.text('it is bad address, try again')
else:
    latitude = location.latitude
    longitude = location.longitude
    st.markdown(latitude)
    st.markdown(longitude)
    if 'borough' in location.raw['address']:
        neighbourhood = location.raw['address']['borough'].replace(
            'London Borough of ', '')
    elif 'city_district' in location.raw['address']:
        neighbourhood = location.raw['address']['city_district'].replace(
            'London Borough of ', '')
    elif 'quarter' in location.raw['address']:
            neighbourhood = location.raw['address']['quarter'].replace(
        'London Borough of ','')
    elif 'suburb' in location.raw['address']:
        neighbourhood = location.raw['address']['suburb'].replace(
            'London Borough of ', '')
    elif 'city' in location.raw['address']:
        neighbourhood = location.raw['address']['city']
    else:
        st.text('address not located')

if neighbourhood is not None :
    st.markdown(neighbourhood)
    url_map = f"https://directingtotheendpoint/maps/?city={city_user}?neigbourhood={neighbourhood}" #create teh endpoint
    # st.markdown(url_map)

############API for the map


# response = requests.get(url).json()
# neighboorhood = response["keywords"]
#     text_to_show = 'the best keywords for '+city_user+' found by our artifical inteligence are : '
#     st.text(text_to_show) #show the text of the  API
#     st.text(city_keywords)

# def density_map_hood(data, neighbourhood, lat_long_hood):
#     lat = lat_long_hood.loc[neighbourhood].latitude
#     lon = lat_long_hood.loc[neighbourhood].longitude
#     m = folium.Map([lat, lon], zoom_start=14, tiles="CartoDB positron")
#     for index, row in data.iterrows():
#         folium.CircleMarker([row['latitude'], row['longitude']],
#                             radius=1,
#                             fill=True,
#                             opacity=0.7).add_to(m)
#     return m
# density plot given neighbourhood

def density_map_hood(data, neighbourhood, lat_long_hood):
    lat = lat_long_hood.loc[neighbourhood].latitude
    lon = lat_long_hood.loc[neighbourhood].longitude
    m = folium.Map([lat, lon], zoom_start=14, tiles="CartoDB positron")
    for row in data.to_numpy():
        folium.CircleMarker([row[1], row[2]],
                            radius=1,
                            fill=True,
                            opacity=0.7).add_to(m)
    return m

data_maps = pd.read_csv("https://storage.googleapis.com/airbnbadvice/data/map_data.csv")
print("data maps loaded")

# @st.cache(suppress_st_warning=True)
def csv_loader(X,neighbourhood):
    X = X[X['neighbourhood_cleansed'] == neighbourhood]
    lat_long_hood = X.groupby("neighbourhood_cleansed").mean()
    maps = density_map_hood(X, neighbourhood,lat_long_hood)
    return folium_static(maps ) #X[["latitude","longitude"]])


# st.map(csv_loader(data_maps,neighbourhood))
csv_loader(data_maps,neighbourhood)

#########################################

if st.button('Artifial Intelligence will compute best fare for your accomodation'):
    url = f"https://airbnbadvice-zktracgm3q-ew.a.run.app/fare_prediction/?latitude={latitude}&longitude={longitude}&accomodates={accomodates}&bedrooms={nb_bedrooms}&beds={nb_beds}&minimum_nights={min_nights}&Entire_home_apt={min_nights}"
    st.text(url)
    response = requests.get(url).json()
    fare_predicted = response['predicted_fare']
    st.text("the predicted price should be ")
    st.text(fare_predicted)

###############
# Rich Rating chart

def neighbourhood_reviews(neighbourhood):
    scores_rating = pd.read_csv('airbnb_advice/data/review_scores.csv')

    labels = ['accuracy', "cleanliness", "location", "communication","value", "checkin"]
    points = len(labels)
    angles = np.linspace(0, 2 * np.pi, points, endpoint=False).tolist()
    angles += angles[:1]
    angles.pop()
    neighbourhood_rating = scores_rating[scores_rating['neighbourhood'] == neighbourhood]
    neighbourhood_rating = neighbourhood_rating.groupby('neighbourhood').median()
    
    def add_to_star_neighbourhood(neighbourhood, color, label=None):
        values = neighbourhood_rating.loc[neighbourhood].tolist()
        values += values[:1]
        del values[0]
        values.pop()
        if label != None:
            ax.plot(angles, values, color=color, linewidth=1, label=label)
        else:
            ax.plot(angles, values, color=color, linewidth=1, label=neighbourhood)
        ax.fill(angles, values, color=color, alpha=0.25)

    ## Create plot object   
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ## Fix axis to star from top
    ax.set_theta_offset(np.pi / 2)

    ax.set_theta_direction(-1)

    # Change the color of the ticks
    ax.tick_params(colors='#222222')


    ax.tick_params(axis='y', labelsize=0)
    # Make the x-axis labels larger or smaller.
    ax.tick_params(axis='x', labelsize=13)


    # Change the color of the circular gridlines.
    ax.grid(color='#AAAAAA')

    # Change the color of the outer circle
    ax.spines['polar'].set_color('#222222')

    # Change the circle background color
    ax.set_facecolor('#FAFAFA')# Add title and legend
    ax.set_title('Comparing Property Ratings', y=1.08)

    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(angles), labels)


    return add_to_star_neighbourhood(neighbourhood, '#1aaf6c', "First Property")

st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(neighbourhood_reviews(neighbourhood))