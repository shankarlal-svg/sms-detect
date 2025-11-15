import streamlit as st
import joblib

# -------------------------------
# Load Model & Vectorizer
# -------------------------------
@st.cache_resource
def load_model_files():
    model = joblib.load("spam_model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_model_files()

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")
st.title("📩 SMS Spam Detector")
st.write("Enter a message below to check if it is Spam or Not Spam.")

# User input
user_input = st.text_area("Type your message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        # Transform input using vectorizer
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        # Display result
        if prediction.lower() == "spam":
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT spam!")
