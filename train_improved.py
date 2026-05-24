# train_improved.py
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
import nltk

# Ensure stopwords are downloaded
try:
    STOPWORDS = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

def load_data(path='data/news_cleaned.csv'):
    df = pd.read_csv(path)
    # Combine title + text if both present
    if 'title' in df.columns and 'text' in df.columns:
        df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    elif 'title' in df.columns and 'text' not in df.columns:
        df['text'] = df['title'].fillna('')
    # ensure label column exists
    if 'label' not in df.columns:
        raise ValueError("Dataset must have a 'label' column.")
    df = df[['text','label']].dropna()
    return df

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)           # remove urls
    text = re.sub(r'[^a-z\s]', ' ', text)          # remove non-letters
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

def main():
    print("=== IMPROVED MODEL TRAINING ===")
    print("Loading dataset...")
    df = load_data('data/news.csv')
    print("Samples:", len(df))
    
    print("Cleaning text...")
    df['text'] = df['text'].astype(str).apply(clean_text)
    
    # Normalize labels to lower-case
    df['label'] = df['label'].astype(str).str.lower().map(lambda s: 'fake' if 'fake' in s else 'real')

    X = df['text'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Vectorizing text with IMPROVED TF-IDF...")
    # IMPROVED: Better vectorizer settings
    vectorizer = TfidfVectorizer(
        max_features=20000,           # More features
        ngram_range=(1, 3),           # Capture phrases (1-3 words)
        min_df=2,                     # Ignore very rare words
        max_df=0.85,             # remove too common words
     stop_words='english'                    
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training IMPROVED Logistic Regression...")
    # IMPROVED: Better model parameters
    model = LogisticRegression(
        max_iter=3000,                # More iterations for convergence
        class_weight='balanced',      # Handle class imbalance
        C=1.2,  
         solver='lbfgs',                     
        random_state=42
    )
    model.fit(X_train_vec, y_train)

    print("Evaluating IMPROVED model...")
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"IMPROVED Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, preds))

    # Save improved artifacts
    import os
    os.makedirs('models', exist_ok=True)
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\n✅ IMPROVED model saved! (overwrote previous files)")

if __name__ == '__main__':
    main()