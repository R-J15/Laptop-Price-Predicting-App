import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image


model = pickle.load(open('model.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))



st.title('Laptop Price')

image = Image.open('resourse\image\laptop.jpeg')
st.image(image, width=500)



brand = st.selectbox('Brand',df["Brand"].unique())
Processor = st.selectbox('Processor',df["Processor"].unique())
RAM = st.selectbox('RAM',df["RAM"].unique())
Storage = st.selectbox('Storage',df["Storage"].unique())
OS = st.selectbox('OS',df["OS"].unique())
Screen_Size = st.selectbox('Screen_Size',df["Screen_Size"].unique())

if st.button('Predict Price'):
    
    query = np.array([brand,Processor,RAM,Storage,OS,Screen_Size])
    query = query.reshape(1, 6)
    st.title("Laptop Price Prediction : ₹" + str(int(np.exp(model.predict(query)[0]))))




  
