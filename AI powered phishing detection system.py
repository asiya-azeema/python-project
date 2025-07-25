# Import necessary libraries
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline

# Load dataset (assuming it's in the correct path)
data_path = r"C:\Users\azoos\Downloads\CEAS_08_extracted\CEAS_08.csv"
df = pd.read_csv(data_path)

# Data Preprocessing

# 1. Handle missing values by filling NaN with empty string (for text fields)
df.fillna('', inplace=True)

# 2. Feature Engineering
# Combine 'subject' and 'body' as they carry important textual information for phishing detection
df['email_text'] = df['subject'] + " " + df['body']

# 3. Feature for URLs (presence of URLs could indicate phishing attempt)
df['has_urls'] = df['urls'].apply(lambda x: 1 if x else 0)

# 4. Convert 'sender' and 'receiver' columns to lowercase (for simplicity)
df['sender'] = df['sender'].str.lower()
df['receiver'] = df['receiver'].str.lower()

# 5. Encode 'label' (Phishing or legitimate) into numeric form (0: legitimate, 1: phishing)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# 6. Keyword-based feature
def contains_suspicious_keywords(text):
    suspicious_keywords = ['urgent', 'offer', 'free', 'claim', 'winner']
    return any(word in text.lower() for word in suspicious_keywords)

df['has_suspicious_keywords'] = df['email_text'].apply(contains_suspicious_keywords)

# Split the dataset into features and target variable
x = df[['email_text', 'sender', 'receiver', 'has_urls', 'has_suspicious_keywords']]  # Feature columns
y = df['label']  # Target variable (phishing or legitimate)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Text Preprocessing and Model Training Pipeline
# Using TF-IDF Vectorizer for text processing and RandomForestClassifier for training
pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english', max_features=1000),
    RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
)

# Train the model using the pipeline
pipeline.fit(X_train['email_text'], y_train)

# Predictions
y_pred = pipeline.predict(X_test['email_text'])

# Evaluate the model
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# To predict on new data:
new_data = ["Urgent, Your account has been compromised. Click this link immediately to secure your account."]

# Call the function and check for phishing
is_phishing = contains_suspicious_keywords(new_data[0])  # Check if suspicious keyword is present in the email text

# Print "Phishing" if the result is 1, otherwise print "Legitimate"
if is_phishing:
    print("Prediction: Phishing")
else:
    print("Prediction: Legitimate")
