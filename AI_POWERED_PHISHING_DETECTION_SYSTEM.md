# 📧 Phishing Email Detection using Machine Learning

This project uses machine learning to classify emails as **phishing** or **legitimate** based on the content, metadata, and presence of suspicious patterns.

## 🧠 Overview

We use the [CEAS 2008 Spam Email Dataset](http://www.ceas.cc/2008) to train a model that analyzes:

- Email subject and body text
- URL presence
- Sender and receiver patterns
- Suspicious keywords like "urgent", "offer", "free", etc.

The model is trained using a **Random Forest Classifier** with a **TF-IDF vectorizer** to handle email text.

---

## 🗂️ Project Structure

```
📁 phishing-email-detector/
├── phishing_email_classifier.py   # Main training and prediction script
├── CEAS_08.csv                    # Email dataset (not included due to size)
├── README.md                      # Project documentation
```

---

## ⚙️ Features

- Handles missing data and text normalization
- Feature engineering for phishing indicators (URLs, keywords)
- Label encoding for target values
- TF-IDF vectorization of email content
- Random Forest classifier with balanced class weights
- Accuracy and classification report evaluation
- Simple prediction on new email data

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn
```

### 2. Prepare Dataset

Make sure the `CEAS_08.csv` file is in the same directory as the script or update the `data_path` in `phishing_email_classifier.py`.

### 3. Run the Script

```bash
python phishing_email_classifier.py
```

---

## 🧪 Example Prediction

```python
new_data = ["Urgent, your account has been compromised. Click here to verify."]
```

This will output:

```
Prediction: Phishing
```

---

## 📊 Sample Output

```
Accuracy:  0.92
Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.91      0.92       240
           1       0.91      0.93      0.92       260

    accuracy                           0.92       500
```

---

## 👩‍💻 Author

**Asiyamath Azeema**  
Dept. of Artificial Intelligence & Data Science  
Bearys Institute of Technology  
USN: 4BP22AD008

---

## 📘 License

This project is for educational purposes only.
