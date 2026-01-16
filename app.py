import streamlit as st
import joblib

# Page config
st.set_page_config(page_title="Spam Detection App", page_icon="📩", layout="centered")

# Load model
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Sidebar
st.sidebar.title("📌 Project Info")
st.sidebar.write("**Project:** Spam Detection using ML")
st.sidebar.write("**Algorithm:** Naive Bayes")
st.sidebar.write("**Feature:** TF-IDF")
st.sidebar.write("**Developer:** Ayush")

# Main Title
st.markdown("<h1 style='text-align: center;'>📩 Spam Detection Web App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Check whether a message is <b>Spam</b> or <b>Not Spam</b></p>", unsafe_allow_html=True)

st.write("---")

# Example buttons
st.subheader("Try Example Messages:")

col1, col2 = st.columns(2)

if col1.button("🎁 Spam Example"):
    st.session_state.msg = "Congratulations! You won a free recharge"

if col2.button("💬 Normal Example"):
    st.session_state.msg = "Are we meeting in college today?"

# Input box
message = st.text_area("✉️ Enter your message:", value=st.session_state.get("msg", ""))

# Predict
if st.button("🔍 Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        msg_vec = vectorizer.transform([message])
        prediction = model.predict(msg_vec)

        if prediction[0] == 1:
            st.error("🚨 This message is **SPAM**")
        else:
            st.success("✅ This message is **NOT SPAM**")

st.write("---")

# Footer
st.markdown(
    "<p style='text-align: center; color: grey;'>ML Project | Streamlit Deployment</p>",
    unsafe_allow_html=True
)
