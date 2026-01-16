import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Page Config ----------------
st.set_page_config(page_title="Spam Detection Pro", page_icon="📧", layout="centered")

# ---------------- Load Model ----------------
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- Session State ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Dark Mode Toggle ----------------
dark = st.sidebar.toggle("🌙 Dark Mode")

if dark:
    bg = "#0e1117"
    card = "#161b22"
    text = "white"
else:
    bg = "#f7f9fc"
    card = "white"
    text = "black"

# ---------------- Custom CSS ----------------
st.markdown(f"""
<style>
.main {{ background-color: {bg}; color: {text}; }}
.card {{
    background-color: {card};
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.15);
}}
.email-box {{
    background-color: #f1f3f5;
    padding: 15px;
    border-radius: 10px;
    font-family: monospace;
}}
.result-spam {{
    background-color: #ffe5e5;
    padding: 15px;
    border-radius: 10px;
    color: #b30000;
    font-weight: bold;
    text-align: center;
}}
.result-ham {{
    background-color: #e6ffea;
    padding: 15px;
    border-radius: 10px;
    color: #006622;
    font-weight: bold;
    text-align: center;
}}
.footer {{
    text-align: center;
    color: gray;
    margin-top: 25px;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar Info ----------------
st.sidebar.title("📊 Model Info")
st.sidebar.write("Algorithm: Naive Bayes")
st.sidebar.write("Feature: TF‑IDF")
st.sidebar.write("Accuracy: **98.2%**")
st.sidebar.write("Dataset: SMS Spam Collection")
st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Developer: Ayush")

# ---------------- Main Card ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>📧 Spam Detection Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI‑powered Email/SMS Spam Classifier</p>", unsafe_allow_html=True)

# Example Buttons
col1, col2 = st.columns(2)
if col1.button("🎁 Load Spam Email"):
    st.session_state.msg = "Congratulations! You won a free iPhone. Click the link now!"

if col2.button("💬 Load Normal Message"):
    st.session_state.msg = "Hey, are we submitting the ML project today?"

# Fake Email UI
st.markdown("### 📬 Email Preview")
message = st.text_area(
    "From: unknown@offers.com\nSubject: Important Message\n\nMessage:",
    value=st.session_state.get("msg", ""),
    height=160
)

# Predict
if st.button("🚀 Analyze Message"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        msg_vec = vectorizer.transform([message])
        prediction = model.predict(msg_vec)

        label = "SPAM" if prediction[0] == 1 else "NOT SPAM"

        st.session_state.history.append((message[:40] + "...", label))

        st.write("")
        if prediction[0] == 1:
            st.markdown("<div class='result-spam'>🚨 RESULT: SPAM MESSAGE</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-ham'>✅ RESULT: NOT SPAM</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Message History ----------------
st.write("## 🕒 Prediction History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history, columns=["Message", "Result"])
    st.dataframe(df, use_container_width=True)
else:
    st.info("No predictions yet.")

# ---------------- Chart Section ----------------
st.write("## 📊 Spam vs Ham Predictions")

if st.session_state.history:
    chart_df = df["Result"].value_counts()
    st.bar_chart(chart_df)

# ---------------- Footer ----------------
st.markdown("<div class='footer'>🔥 Premium ML Project UI | Streamlit App</div>", unsafe_allow_html=True)
