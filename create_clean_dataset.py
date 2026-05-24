# create_clean_dataset.py
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

# Load original raw data
true_df = pd.read_csv('data/True.csv')
fake_df = pd.read_csv('data/Fake.csv')

# Add labels
true_df['label'] = 'real'
fake_df['label'] = 'fake'

# Combine
df = pd.concat([true_df, fake_df], ignore_index=True)

# Clean the text PROPERLY
print("Cleaning text data...")
df['text'] = df['text'].apply(clean_text)

# Shuffle and save
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('data/news_cleaned.csv', index=False)

print(f"Saved cleaned dataset with {len(df)} samples")
print("Sample of cleaned text:")
print(df['text'].iloc[0][:200] + "...")