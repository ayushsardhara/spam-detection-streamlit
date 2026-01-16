import streamlit as st
import joblib

st.set_page_config(page_title="Spam Detection App")

st.title("📩 Spam Detection using Machine Learning")
st.write("Enter a message to check whether it is Spam or Not Spam")

# Load trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Input box
message = st.text_area("✉️ Type your message here:")

# Predict button
if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message")
    else:
        msg_vec = vectorizer.transform([message])
        prediction = model.predict(msg_vec)

        if prediction[0] == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT SPAM")
