import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------
# Load artifacts
# -------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------
# Text preprocessing (same logic)
# -------------------------
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = "".join(c for c in text if not c.isdigit())
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = " ".join(
        w for w in text.split() if w not in stopwords.words("english")
    )
    lemmatizer = WordNetLemmatizer()
    text = " ".join(lemmatizer.lemmatize(w) for w in text.split())
    return text

# -------------------------
# Streamlit UI
# -------------------------
st.title("Tweet Sentiment Analysis")
st.write("Enter a tweet to predict sentiment")

user_input = st.text_area("Tweet text")

if st.button("Predict Sentiment"):
    cleaned = clean_text(user_input)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("😊 Positive Sentiment")
    else:
        st.error("😠 Negative Sentiment")
