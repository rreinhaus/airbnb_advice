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
import plotly.express as px
import seaborn as sns
import plotly.figure_factory as ff
import time

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

#density plot given neighbourhood


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

# @st.cache(suppress_st_warning=True)
def csv_loader(X,neighbourhood):
    X = X[X['neighbourhood_cleansed'] == neighbourhood]
    lat_long_hood = X.groupby("neighbourhood_cleansed").mean()
    maps = density_map_hood(X, neighbourhood,lat_long_hood)
    return folium_static(maps) #X[["latitude","longitude"]])

###############
# Rich Rating chart

def neighbourhood_reviews(neighbourhood):
    scores_rating = pd.read_csv('https://storage.googleapis.com/airbnbadvice/data/review_scores.csv')

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

def occup_per_year(price_df, neighbourhood):

    occup_hood = price_df.groupby("neighbourhood_cleansed").median()[[

        'occupancy_month', 'occupancy_year'

    ]].reset_index()

    occupied = occup_hood[(

        occup_hood['neighbourhood_cleansed'] == neighbourhood)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

    fig.suptitle('Percentages of occupancy', fontsize=20)

    to_plot_month = [

        float(occupied['occupancy_month']),

        1 - float(occupied['occupancy_month'])

    ]

    mylabels_month = [

        'occupied' + '\n' +

        str(round(float(occupied['occupancy_month']) * 100, 2)) + '%', 'vacant'

    ]

    myexplode = [0.1, 0]

    ax1.pie(

        to_plot_month,

        labels=mylabels_month,

        explode=myexplode,

        radius=1.3,

        labeldistance=0.5,

        #rotatelabels=True,

        textprops=dict(rotation_mode='anchor', va='center', ha='left'),

    )

    ax1.set_title('Monthly occupancy', y=1.08)

    to_plot_year = [

        float(occupied['occupancy_year']),

        1 - float(occupied['occupancy_year'])

    ]

    mylabels_year = [

        'occupied' + '\n' +

        str(round(float(occupied['occupancy_year']) * 100, 2)) + '%', 'vacant'

    ]

    ax2.pie(

        to_plot_year,

        labels=mylabels_year,

        explode=myexplode,

        radius=1.3,

        labeldistance=0.5,

        textprops=dict(rotation_mode='anchor', va='center', ha='left'),

    )

    ax2.set_title('Yearly occupancy', y=1.08)

    return fig, to_plot_month[0], to_plot_year[0]

def draw_plot(df, column):
    '''
    Returns a scatter plot
    '''
    fig = px.histogram(df, x=column, nbins=10)

    return fig

def keywords(neighbourhood):
    for char in neighbourhood:
        if char == ' ':
            neighbourhood = neighbourhood.replace(" ", "%20")

    df_comments = pd.read_csv(f'https://storage.googleapis.com/airbnbadvice/data/keywords/comments_keywords/{neighbourhood}_comments.csv')
    df_neighbourhood = pd.read_csv(f'https://storage.googleapis.com/airbnbadvice/data/keywords/neighbourhood_keywords/{neighbourhood}_neighbourhood.csv')
    df_description = pd.read_csv(f'https://storage.googleapis.com/airbnbadvice/data/keywords/description_keywords/{neighbourhood}.csv')

    # list of words
    keys_comments = []
    keys_neighbourhoods = []
    keys_description = []

    for key in range(5):
        keys_comments.append(df_comments['keywords'][key])
        keys_neighbourhoods.append(df_neighbourhood['keywords'][key])
        keys_description.append(df_description['keywords'][key])

    comments = [item for item in keys_comments if item not in keys_neighbourhoods]
    neigh = [item for item in keys_neighbourhoods if item not in keys_comments]
    des = [item for item in keys_description if item not in keys_neighbourhoods]

    col1, col2, col3 = st.columns(3)
    comment_col = [col1, col2, col3]

    for i, val in enumerate(comments):
        first_comments = comment_col[i].metric(str(i+1)+': comments', comments[i])

    col1_n, col2_n, col3_n = st.columns(3)

    neigh_col = [col1_n, col2_n, col3_n]

    for i, val in enumerate(neigh):
        first_n = neigh_col[i].metric(str(i+1)+': Neighbourhood', neigh[i])

    return first_comments, first_n

###################################### Streamlit Start ################################

neighbourhood = 'Camden'
latitude = 51.51988089870245
longitude = -0.041186831798636546

data_maps = pd.read_csv("https://storage.googleapis.com/airbnbadvice/data/map_data.csv")

st.set_page_config(
            page_title="Airbnb Advice", # => Quick reference - Streamlit
            page_icon="🏠",
            layout="wide", # wide
            initial_sidebar_state="auto") # collapsed

st.markdown('''
# AIRBNB ADVICE

The aim of the application is to provide Airbnb hosts and potential hosts insights about their neighbourhood.

We also try to help to optimize listings by providing price per night prediction and auto-generated titles and descriptions''')

st.markdown('---')

#####  collection of the data form the USER in the SIDEBAR
country=st.sidebar.text_input('Select a country','UK')
city_user = st.sidebar.selectbox('Select a city',  ["London",""] )
address = st.sidebar.text_input("Address", "Your street name and number")
full_adress = address + city_user + country

entire_home = st.sidebar.checkbox("Are you renting entire home?", value = False ) # binary does the user rent the full accomodation or not
nb_bedrooms = st.sidebar.slider("Number of rooms", 1,10,2) #the number of room
nb_beds = st.sidebar.slider("Number of Beds",1,10,2) # the number of beds
min_nights = st.sidebar.slider("Minimum nights that people have to stay", 1,7,1)
accomodates = int(st.sidebar.number_input('How many guests can you accomodate?' , min_value=0, value=5, step=1 ))
navi = st.sidebar.radio("Data Sections To Display", ('Neighbourhood Stats','Title & Describtions'))

# Getting pick up location as address and transforming to coordinates
#neighbourhood = None
loc = Nominatim(user_agent= "GetLoc" )
geocode = RateLimiter(loc.geocode, min_delay_seconds=1)
location = loc.geocode(address +","+city_user+","+ country,addressdetails=True)

if location is None:
    st.text('Address is missing! Please enter street name in the sidebar')
else:
    latitude = location.latitude
    longitude = location.longitude
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

# DataFrames for the code
data = pd.read_csv(
    'https://storage.googleapis.com/airbnbadvice/data/superhost.csv')
price_df = pd.read_csv(
    'https://storage.googleapis.com/airbnbadvice/data/price.csv')
price_df['occupancy_month'] = price_df['days_booked_month'] / 31
ame = pd.read_csv(
    'https://storage.googleapis.com/airbnbadvice/data/ame_final.csv')

# PRICE PREDICTION

if st.sidebar.button(
        'Find out the best price per night for your property and potential revenue'):
    url = f"https://airbnbadvice-zktracgm3q-ew.a.run.app/fare_prediction/?latitude={latitude}&longitude={longitude}&accomodates={accomodates}&bedrooms={nb_bedrooms}&beds={nb_beds}&minimum_nights={min_nights}&Entire_home_apt={min_nights}"
    # st.text(url)
    my_bar = st.sidebar.progress(0)
    response = requests.get(url).json()

    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)

    fare_predicted = response['predicted_fare']
    st.sidebar.markdown(
        f"The price per nigth should be **{round(fare_predicted,2)} £** ")

    monthly_revenue = occup_per_year(
            price_df, neighbourhood)[1] * 30.5 * fare_predicted

    yearly_revenue = occup_per_year(
        price_df, neighbourhood)[2] * 365 * fare_predicted

    st.sidebar.markdown(
        f'''If you achieve the highest occupancy rate of your area,
    your potential revenue is of £ {round(monthly_revenue)} per month and
    £ {round(yearly_revenue)} per year.''')

    # if fare_predicted is not None:

    #     monthly_revenue = occup_per_year(
    #         price_df, neighbourhood)[1] * 30.5 * fare_predicted

    #     yearly_revenue = occup_per_year(
    #         price_df, neighbourhood)[2] * 365 * fare_predicted

    #     st.markdown(f'''If you achieve the average occupancy rate of your area,
    #     your potential revenue is of £{round(monthly_revenue,2)} per month and
    #     £{round(yearly_revenue,2)} per year.''')

    # else:

    #     st.text('''Please run the model above first.''')







if navi == 'Title & Describtions':
    st.markdown("""
            # Listing Optimization Zone

            ### Auto-Generated Airbnb title

            Please give two - three words that describes your property
        """)

    words = st.text_input('Property Describtion')
    if words:
        url = f"https://airbnbadvice-zktracgm3q-ew.a.run.app/announcement?keywords1={words}"
        response_announce = requests.get(url).json()
        announce_predicted = response_announce['announce']
        st.text(words + ' - ' + announce_predicted)

    st.markdown('---')
    st.markdown("""
            ### The top 3 Superhost comment & neighbourhood overview keywords

            These are the keywords specific to your area that unsupervised machine learning model came up with ;)
        """)
    keywords(neighbourhood)

    st.markdown("""

            ### 🚧 work in progress 🚧

            Next week we will try to demo also full description generator 😉

        """)
else:
    st.markdown(f'### Neighbourhood Stats - {neighbourhood}')
    st.markdown('---')
    st.markdown('**Top 10 Amenities**')

    ame_col1, ame_col2, ame_col3, ame_col4, ame_col5 = st.columns(5)
    with ame_col1:
        st.markdown('wifi')
        st.markdown('Essentials')

    with ame_col2:
        st.markdown('Kitchen')
        st.markdown('Heating')

    with ame_col3:
        st.markdown('Smoke Alarm')
        st.markdown('Long term stays allowed')

    with ame_col4:
        st.markdown('Washer')
        st.markdown('Hanger')

    with ame_col5:
        st.markdown('Iron')
        st.markdown('Hair Dryer')

    st.markdown('---')

    # Amenities

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Number of Amenities from Top 10",
                value=f"{round(ame[ame['neighbourhood_cleansed'] == neighbourhood].iloc[0]['total_top10'])}",
                delta=None,
                delta_color="normal")
    with col2:
        st.metric(label='Properties rented as Entire Home',
                value=f"{round(ame[ame['neighbourhood_cleansed'] == neighbourhood].iloc[0]['entire_home_percent'] *100,2)}%",
                delta=None,
                delta_color="normal")
    with col3:
        st.metric(
            label='Properties rented as Private Room',
            value=
            f"{round(ame[ame['neighbourhood_cleansed'] == neighbourhood].iloc[0]['private_room_percent'] *100,2)}%",
            delta=None,
            delta_color="normal")
    with col4:
        st.metric(
            label='Properties rented as Share Room',
            value=
            f"{round(ame[ame['neighbourhood_cleansed'] == neighbourhood].iloc[0]['shared_room_percent'] *100,2)}%",
            delta=None,
            delta_color="normal")

    st.markdown('---')

    expander = st.expander("Density Map - More Info")

    expander.markdown("""The Neighbourhood Density Map.

            The map presents each property within your neighbourhood and where it is located.

            The aim of the map is to see where the most properties are located in your specific area and how many there are.

            """)
    c1, c2, c3 = st.columns([1, 3, 1])

    with c2:
        csv_loader(data_maps,neighbourhood)

    # Density Map & Price Plots
    chart1, chart2 = st.columns(2)
    with chart1:
        occupancy_expand = st.expander("Occupancy Rates - More Info")
        occupancy_expand.markdown("""Occupancy Rate Target Chart.

                The chart showcases based on each listed property avaible and booked previous nights what is the aimed occupancy rate for each property.

                P.S. The property occupancy rates are not guaranteed and this is not financial advise by any means.
                """)
        # Occupancy Plot

        occup_plot = draw_plot(price_df[['occupancy_month']],
                               'occupancy_month')
        occup_plot.update_traces(marker_line_width=0.5,
                                 marker_line_color="white")
        occup_plot.update_layout(bargap=0.1)
        st.plotly_chart(occup_plot, use_container_width=True)
        # with st.container():
        #     csv_loader(data_maps,neighbourhood)



    with chart2:
        # Price Plot
        price_expand = st.expander("Property Price - More Info")

        price_expand.markdown("""Property Price Distribution.

            The chart showcases how the properties are priced within your neighbourhood.

            Please take into consideration for better representation we have taken out the outliers.

            """)
        Q1 = price_df.price.quantile(0.25).round(3)
        Q3 = price_df.price.quantile(0.75).round(3)
        IQR = Q3 - Q1
        outlier = Q3 + 1.5*IQR
        price_plot = px.histogram(price_df[price_df['price'] <= outlier], x='price', nbins=10)
        price_plot.update_traces(marker_line_width=0.5,
                                 marker_line_color="white")
        price_plot.update_layout(bargap=0.1)
        st.plotly_chart(price_plot, use_container_width=True)
        # st.write(price_plot)

    # Superhost & Revenue
    chart3, chart4 = st.columns(2)
    with chart3:
        # Neighbourhood Superhost Plot
        superhost_expand = st.expander("Superhost status - More Info")

        superhost_expand.markdown("""Superhost status chart.

            The chart showcases how many of the total neighbourhood property owners are superhosts.
            """)
        fig = px.histogram(data[data['neighbourhood_cleansed']== neighbourhood], x="host_is_superhost")
        fig.update_traces(marker_line_width=0.5,
                          marker_line_color="white")
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
        # st.write(fig)

    with chart4:
        # Revenue plot
        rev_expand = st.expander("Monthly revenues - More Info")

        rev_expand.markdown("""Monhtly revenue chart.

            The chart showcases based on each property occupancy rate, what is the distribution of airbnb property potential revenue targets.

            P.S. The property revenues per month are not guaranteed and this is not financial advise by any means.
            """)

        group_labels = ['pot_rev_month']
        rev_plot = px.histogram(price_df[['pot_rev_month']], x='pot_rev_month', nbins=10)
        #rev_plot = ff.create_distplot([price_df['pot_rev_month']],group_labels, bin_size=.10, curve_type='normal')
        rev_plot.update_traces(marker_line_width=0.5,
                                 marker_line_color="white")
        rev_plot.update_layout(bargap=0.1)
        st.plotly_chart(rev_plot, use_container_width=True)
        # st.write(rev_plot)

    # # Ratings & Occupancy
    # chart5, chart6 = st.columns(2)
    # with chart5:
    #     # Occupancy Plot
    #     occupancy_expand = st.expander("Occupancy Rates - More Info")
    #     occupancy_expand.markdown("""Occupancy Rate Target Chart.

    #             The chart showcases based on each listed property avaible and booked previous nights what is the aimed occupancy rate for each property.

    #             P.S. The property occupancy rates are not guaranteed and this is not financial advise by any means.
    #             """)
    #     occup_plot = draw_plot(price_df[['occupancy_month']], 'occupancy_month')
    #     occup_plot.update_traces(marker_line_width=0.5,
    #                              marker_line_color="white")
    #     occup_plot.update_layout(bargap=0.1)
    #     st.plotly_chart(occup_plot, use_container_width=True)
    #     # st.write(occup_plot)

    # with chart6:
    #     # Neigbourhood Rating Plot
    #     st.set_option('deprecation.showPyplotGlobalUse', False)
    #     st.pyplot(neighbourhood_reviews(neighbourhood))

    # Potential Revenue Based On your apartment plot
    # if st.button('What is your potential revenue?'):
    #     url = f"https://airbnbadvice-zktracgm3q-ew.a.run.app/fare_prediction/?latitude={latitude}&longitude={longitude}&accomodates={accomodates}&bedrooms={nb_bedrooms}&beds={nb_beds}&minimum_nights={min_nights}&Entire_home_apt={min_nights}"
    #     response = requests.get(url).json()
    #     fare_predicted = response['predicted_fare']
    #     if fare_predicted is not None:

    #         monthly_revenue = occup_per_year(

    #             price_df, neighbourhood)[1] * 30.5 * fare_predicted

    #         yearly_revenue = occup_per_year(

    #             price_df, neighbourhood)[2] * 365 * fare_predicted

    #         st.text(f'''If you achieve the average occupancy rate of your area,
    #         your potential revenue is of £{round(monthly_revenue,2)} per month and
    #         £{round(yearly_revenue,2)} per year.''')

    #         st.pyplot(occup_per_year(price_df, neighbourhood)[0])

    #     else:

    #         st.text('''Please run the model above first.''')

    # The NLP Part
    #if st.checkbox('Do you want to optimize your listing?'):



    # # London Keywords
    # if st.button('Top Keywords Used by Superhosts in London'):
    #     url = "https://airbnbadvice-zktracgm3q-ew.a.run.app/keywords/?city="+city_user
    #     response = requests.get(url).json()
    #     city_keywords = response["keywords"]
    #     text_to_show = 'the best keywords for '+city_user+' found by our artifical inteligence are : '
    #     st.text(text_to_show) #show the text of the  API
    #     st.text(city_keywords)
    #     #st.table(city_keywords)
