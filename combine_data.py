#combine_data.py
import pandas as pd

# Read the two separate files
true_df = pd.read_csv('data/True.csv')
fake_df = pd.read_csv('data/Fake.csv')

# Add labels
true_df['label'] = 'real'
fake_df['label'] = 'fake'

# Combine them into one DataFrame
combined_df = pd.concat([true_df, fake_df], ignore_index=True)

# Shuffle the dataset to mix real and fake news
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the combined file
combined_df.to_csv('data/news.csv', index=False)

print("Dataset successfully created as 'data/news.csv'")
print(f"Total samples: {len(combined_df)}")