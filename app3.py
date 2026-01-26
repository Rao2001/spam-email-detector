import streamlit as st
import pickle

# -------------------------------
# Load saved model & vectorizer
# -------------------------------
@st.cache_data
def load_model():
    with open("modelspam.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizerspam.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📧 Spam Email Detector")
st.write("Enter an email text below to see if it's SPAM or NOT SPAM:")

user_input = st.text_area("Email text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        if prediction == 1:
            st.error("🚫 SPAM")
        else:
            st.success("✅ NOT SPAM")