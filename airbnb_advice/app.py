import streamlit as st

st.markdown('''
# AIR BNB ADVIC€
## Welcolme to our amazing app
Richard, Nicolas, Christe

''')

st.markdown('''
#Thanks to provide the data in the inbox below so Artificial Intelligence can predict the TAXI FARE  : 
''')

st.slider("number or rooms", 1,10,2)

st.markdown(""" <style> .font {
font-size:50px ; font-family: 'Cooper Black'; color: #FF9633;} 
</style> """, unsafe_allow_html=True)

st.markdown('<p class="font">Guess the object Names</p>', unsafe_allow_html=True)
