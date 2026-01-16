import streamlit as st
import joblib

# Page config
st.set_page_config(page_title="Spam Detection App", page_icon="📩", layout="centered")

# Load model
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---- CUSTOM CSS ----
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.card {
    background-color: #161b22;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.4);
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #9ba3af;
}
</style>
""", unsafe_allow_html=True)

# ---- SIDEBAR ----
st.sidebar.title("📌 Project Details")
st.sidebar.markdown("**Project:** Spam Detection using ML")
st.sidebar.markdown("**Algorithm:** Naive Bayes")
st.sidebar.markdown("**Feature Extraction:** TF‑IDF")
st.sidebar.markdown("**Deployment:** Streamlit Cloud")
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 Developed by Ayush")

# ---- MAIN CARD ----
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("<div class='title'>📩 Spam Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Check whether a message is Spam or Not Spam</div>", unsafe_allow_html=True)
st.write("")

# Example buttons
col1, col2 = st.columns(2)
if col1.button("🎁 Try Spam Example"):
    st.session_state.msg = "Congratulations! You won a free gift card. Click now!"
if col2.button("💬 Try Normal Example"):
    st.session_state.msg = "Hey, are we meeting for the project today?"

# Input
message = st.text_area("✉️ Enter your message:",
                       value=st.session_state.get("msg", ""),
                       height=120)

# Predict
if st.button("🚀 Predict"):
    if message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        msg_vec = vectorizer.transform([message])
        prediction = model.predict(msg_vec)

        if prediction[0] == 1:
            st.error("🚨 RESULT: This message is **SPAM**")
        else:
            st.success("✅ RESULT: This message is **NOT SPAM**")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    "<p style='text-align:center; color:#9ba3af;'>Machine Learning Project | Streamlit Web App</p>",
    unsafe_allow_html=True
)
