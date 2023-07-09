import numpy as np
import pickle
import pandas as pd
import streamlit as st
import base64
from PIL import Image


pickle_in = open("classifier.pkl", "rb")
regressor = pickle.load(pickle_in)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )



def welcome():
    return "Welcome All"

def weather_pred(LandAvgTemp, LandMaxTemp, LandMinTemp):
    prediction = regressor.predict([[LandAvgTemp, LandMaxTemp, LandMinTemp]])
    print(prediction)
    return prediction


def main():
    # st.title("Weather Prediction App")
    temp_html = """
        <div style="background-color:#0068c9;padding:10px">
        <h2 style="color:white;text-align:center;"> Weather Prediction App </h2>
    """
    html_temp = """
        <div style="background-color:black;padding:10px">
        <h2 style="color:white;text-align:center;"> Enter the parameters asked down below to predict weather </h2>
        </div>
    """
    st.markdown(temp_html, unsafe_allow_html=True)
    st.markdown(html_temp, unsafe_allow_html=True)
    add_bg_from_local('weather.png')
    LandAvgTemp = st.number_input("Enter Land Average Temperature", value = 0.0)
    LandMaxTemp = st.number_input("Enter Land Maximum Temperature", value = 0.0)
    LandMinTemp = st.number_input("Enter Land Minimum Temperature", value = 0.0)
    result = weather_pred(LandAvgTemp, LandMaxTemp, LandMinTemp)
    if st.button("Predict"):
        st.success('The weather is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")


if __name__ == '__main__':
    main()


