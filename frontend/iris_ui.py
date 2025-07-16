import streamlit as st
import requests

st.title("Iris Species Predictor")
st.write('Enter measuresments below:')

sepal_length = st.slider('Sepal length',0.0,10.0,5.0)
sepal_width = st.slider('Sepal width',0.0,10.0,5.0)
petal_length = st.slider('Petal length',0.0,10.0,5.0)
petal_width = st.slider('Petal width',0.0,10.0,5.0)

if st.button('Predict'):
    payload={
            'sepal_length':sepal_length,
            'sepal_width':sepal_width,
            'petal_length':petal_length,
            'petal_width':petal_width,
    }
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if response.status_code==200:
            result = response.json()
            st.success(f"🌼 Predicted Species: **{result['Prediction']}**")
        else:
            st.error(f"❌ API Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Could not connect to FastAPI server. Is it running?")


