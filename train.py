# train.py
import pandas as pd
import numpy as np
import re
import pickle
import os
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK stopwords if not present
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def load_data(path='data/news.csv'):
    df = pd.read_csv(path)
    # Combine title + text if both present
    if 'title' in df.columns and 'text' in df.columns:
        df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    elif 'title' in df.columns and 'text' not in df.columns:
        df['text'] = df['title'].fillna('')
    # Ensure label column exists
    if 'label' not in df.columns:
        raise ValueError("Dataset must have a 'label' column.")
    df = df[['text', 'label']].dropna()
    return df

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)           # remove URLs
    text = re.sub(r'[^a-z\s]', ' ', text)         # remove non-letters
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

def main():
    print("Loading dataset...")
    df = load_data('data/news.csv')
    print(f"Total samples: {len(df)}")

    print("Cleaning text...")
    df['text'] = df['text'].astype(str).apply(clean_text)

    print("Normalizing labels...")
    # Convert labels to 'fake' or 'real'
    df['label'] = df['label'].astype(str).str.lower().map(lambda s: 'fake' if 'fake' in s else 'real')

    X = df['text'].values
    y = df['label'].values

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("Evaluating model...")
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, preds))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, preds))

    # Save artifacts
    os.makedirs('models', exist_ok=True)
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\nSaved vectorizer.pkl and model.pkl in 'models/' folder.")

if __name__ == '__main__':
    main()
