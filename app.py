import streamlit as st
import pickle
import re
import string

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ------------------- BACKGROUND STYLE -------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right,  #cce7ff, #f8f9fa);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- LOAD MODEL -------------------
model = pickle.load(open('best_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# ------------------- TEXT CLEANING -------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# ------------------- PREDICTION FUNCTION -------------------
def predict_news(news_text):
    cleaned = clean_text(news_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    
    # Confidence score (if available)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vectorized)[0][1]
    else:
        prob = None

    return prediction[0], prob

# ------------------- UI -------------------

# Title
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>📰 Fake News Detection System</h1>",
    unsafe_allow_html=True
)

st.write("### 🔍 Enter news text below to check whether it is real or fake:")

# Input box
news_input = st.text_area("", height=150)

# Button
if st.button("🚀 Check News"):
    if news_input.strip() != "":
        prediction, prob = predict_news(news_input)

        if prediction == 1:
            st.success("✅ This is REAL NEWS")
        else:
            st.error("❌ This is FAKE NEWS")

        # Show confidence score
        if prob is not None:
            st.info(f"🔢 Confidence: {prob*100:.2f}%")

    else:
        st.warning("⚠️ Please enter some text!")