import streamlit as st
import joblib

st.title("SMS Spam Detection")
st.write("Type your SMS message below and find out if it's Spam or Ham!")

# Load model & vectorizer
model = joblib.load('spam_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

sms_input = st.text_area("Enter your message:")

if st.button("Predict"):
    if sms_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        sms_count = vectorizer.transform([sms_input])
        prediction = model.predict(sms_count)
        label = "Spam" if prediction[0]==1 else "Ham"
        st.success(f"Prediction: {label}")
