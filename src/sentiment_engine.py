import os
import re
import joblib
import pandas as pd
import numpy as np
from lime.lime_text import LimeTextExplainer

# MODEL LOADING

def load_models():
    """Finds the models folder regardless of where this script is called from."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        
        model_path = os.path.join(root_dir, 'models', 'sentiment_model.pkl')
        vec_path = os.path.join(root_dir, 'models', 'tfidf_vectorizer.pkl')
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        return model, vectorizer
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

model, vectorizer = load_models()

def clean_text(text):
    """Sanitizes raw review text."""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# PREDICTION LOGIC

def predict_sentiment(text):
    """Predicts sentiment and applies the 40%-60% Mixed bounds."""
    if model is None or vectorizer is None:
        return "🔴 Error", 0.0

    vec = vectorizer.transform([clean_text(text)])
    pos_prob = model.predict_proba(vec)[0][1]
    
    if pos_prob < 0.40: return "🔴 Disappointed", pos_prob
    elif pos_prob > 0.60: return "🟢 Satisfied", pos_prob
    else: return "🟡 Mixed Feelings", pos_prob

# 3. EXPLAINABLE AI (LIME)

explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])

def predict_pipeline(texts):
    """LIME requires a pipeline function that takes raw text and outputs probabilities."""
    if model is None or vectorizer is None:
        return np.zeros((len(texts), 2)) 
    
    cleaned_texts = [clean_text(t) for t in texts]
    vecs = vectorizer.transform(cleaned_texts)
    return model.predict_proba(vecs)

def get_lime_explanation(text, num_features=10):
    """Uses LIME to extract word weights for text highlighting."""
    if not model: return []
    exp = explainer.explain_instance(text, predict_pipeline, num_features=num_features)
    return exp.as_list()