import pandas as pd
import re
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def clean_text(text):
    # Cleaning Datasets
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text) 
    text = re.sub(r'[^a-z\s]', '', text)  
    return re.sub(r'\s+', ' ', text).strip() 

def predict_custom_sentiment(text, model, vectorizer):
    
    vec = vectorizer.transform([clean_text(text)])
    pos_prob = model.predict_proba(vec)[0][1] 
    
    if pos_prob < 0.40:
        return "User is Disappointed", pos_prob
    elif pos_prob > 0.60:
        return "User is Satisfied", pos_prob
    else:
        return "User has Mixed Feelings", pos_prob

if __name__ == "__main__":
    print("="*40)
    print("COMPILING ML MODEL")
    print("="*40)

    # DATA PREPROCESSING

    print("Loading IMDB 50k Dataset...")
    master_df = pd.read_csv("../datasets/IMDB_Dataset.csv")
    master_df['clean_review'] = master_df['review'].apply(clean_text)
    
    # Map text labels to binary integers for machine learning
    master_df['label'] = master_df['sentiment'].map({'positive': 1, 'negative': 0})
    master_df = master_df.dropna(subset=['label'])

    # 80/20 Train-Test Split with Stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        master_df['clean_review'], 
        master_df['label'], 
        test_size=0.20, 
        random_state=42, 
        stratify=master_df['label']
    )

    # FEATURE EXTRACTION (TF-IDF)

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=40000, 
        ngram_range=(1, 2), 
        stop_words=None, # Keeping stop words captures context like "not good"
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # MODEL TRAINING

    print("Training Calibrated LinearSVC...")
 
    base_model = LinearSVC(C=0.5, max_iter=10000, random_state=42)
    model = CalibratedClassifierCV(base_model, cv=5)
    model.fit(X_train_vec, y_train)

    # ACCURACY TEST
   
    print("Accuracy Testing...")
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f" -> Base Model Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)']))

    # EXPORT MODELS

    print("Exporting Model and Vectorizer to /models/...")
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, '../models/sentiment_model.pkl')
    joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')
    print("Build Complete.\n")

    # TESTING
   
    print("="*40)
    print("TESTING...")
    print("="*40)
    
    test_cases = [
        "absolute masterpiece of garbage",  # Testing Sarcasm
        "good acting but terrible plot",    # Testing Mixed/Negative pull
        "a solid one time watch if you have literally nothing else to do", # Testing True Mixed
        "great visuals, shame about the script" # Testing Mixed
    ]
    
    for t in test_cases:
        label, conf = predict_custom_sentiment(t, model, vectorizer)
        print(f'Review: "{t}"')
        print(f'Result: {label} ({conf*100:.1f}% Positive)')
        print("-" * 40)