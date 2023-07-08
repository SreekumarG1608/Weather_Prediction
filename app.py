import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("classifier.pkl","rb")
regressor=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def weather_pred(LandAvgTemp,LandMaxTemp,LandMinTemp):

   
    prediction=regressor.predict([[LandAvgTemp,LandMaxTemp,LandMinTemp]])
    print(prediction)
    return prediction



def main():
    st.title("Weather Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    LandAvgTemp = st.text_input("Enter Land Average Temperature","Type Here")
    LandMaxTemp = st.text_input("Enter Land Maximum Temperature","Type Here")
    LandMinTemp = st.text_input("Enter Land Minimum Temperature","Type Here")
    result=weather_pred(LandAvgTemp,LandMaxTemp,LandMinTemp)
    if st.button("Predict"):
        st.success('The weather is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()