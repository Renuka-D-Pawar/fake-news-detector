# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import os
import re
from nltk.corpus import stopwords
import nltk
import requests
from googletrans import Translator

app = Flask(__name__, static_folder='static', template_folder='templates')

# =========================
# LOAD MODEL & VECTORIZER
# =========================
MODEL_PATH = os.path.join('models', 'model.pkl')
VECT_PATH = os.path.join('models', 'vectorizer.pkl')

print("Loading model and vectorizer...")

with open(VECT_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

print("Model loaded successfully!")

# =========================
# LOAD STOPWORDS
# =========================
try:
    STOPWORDS = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

# =========================
# TRANSLATOR
# =========================
translator = Translator()

# =========================
# NEWS API
# =========================
API_KEY = "YOUR_NEWS_API_KEY"

def fetch_latest_news(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={API_KEY}"

    response = requests.get(url)
    data = response.json()

    articles = []

    for article in data.get("articles", [])[:5]:
        title = article.get("title", "")
        desc = article.get("description", "")
        articles.append(title + " " + desc)

    return articles

# =========================
# TRANSLATION FUNCTION
# =========================
def translate_to_english(text):
    try:
        translated = translator.translate(text, dest='en')
        return translated.text
    except Exception as e:
        print("Translation Error:", e)
        return text

# =========================
# TEXT CLEANING
# =========================
def clean_user_text(text):

    text = str(text).lower()

    # remove urls
    text = re.sub(r'http\S+', ' ', text)

    # remove special characters
    text = re.sub(r'[^a-z\s]', ' ', text)

    tokens = text.split()

    # remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    return " ".join(tokens)

# =========================
# HOME PAGE
# =========================
@app.route('/')
def index():
    return render_template('index.html')

# =========================
# PREDICTION API
# =========================
@app.route('/predict', methods=['POST'])
def predict():

    try:
        data = request.json

        text = data.get('text', '').strip()

        if not text:
            return jsonify({
                'error': 'Please enter some text'
            }), 400

        # =========================
        # TRANSLATE TO ENGLISH
        # =========================
        translated_text = translate_to_english(text)

        # =========================
        # CLEAN TEXT
        # =========================
        cleaned_text = clean_user_text(translated_text)

        if len(cleaned_text.split()) < 3:
            return jsonify({
                'error': 'Please enter more meaningful text'
            }), 400

        # =========================
        # VECTORIZE
        # =========================
        x_vec = vectorizer.transform([cleaned_text])

        # =========================
        # PREDICT
        # =========================
        prediction = model.predict(x_vec)[0]

        # =========================
        # CONFIDENCE SCORE
        # =========================
        probabilities = model.predict_proba(x_vec)[0]

        class_index = list(model.classes_).index(prediction)

        confidence = float(probabilities[class_index])

        # =========================
        # FETCH REAL-TIME NEWS
        # =========================
        latest_news = fetch_latest_news(cleaned_text[:50])

        # =========================
        # SIMPLE MATCH CHECK
        # =========================
        match_found = False

        for news in latest_news:

            for word in cleaned_text.split()[:5]:

                if word in news.lower():
                    match_found = True
                    break

        # =========================
        # FINAL RESPONSE
        # =========================
        return jsonify({

            'original_text': text,

            'translated_text': translated_text,

            'prediction': prediction,

            'confidence': confidence,

            'real_time_match': match_found,

            'latest_news': latest_news[:3]

        })

    except Exception as e:

        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    app.run(debug=True, port=5000)