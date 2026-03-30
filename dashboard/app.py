import streamlit as st
import pandas as pd
import sys
import os
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src import sentiment_engine
import visualizations

st.set_page_config(page_title="Consensus AI", page_icon="📊", layout="wide")

def render_highlighted_text(text, lime_weights):
    highlighted_text = text
    lime_weights.sort(key=lambda x: len(x[0]), reverse=True)
    
    for word, weight in lime_weights:
        if weight > 0: 
            alpha = min(1.0, weight * 5) 
            color = f"rgba(40, 167, 69, {alpha})"
        else: 
            alpha = min(1.0, abs(weight) * 5)
            color = f"rgba(220, 53, 69, {alpha})"
            
        pattern = r'\b' + re.escape(word) + r'\b'
        replacement = f'<span style="background-color: {color}; border-radius: 3px; padding: 0 2px;">{word}</span>'
        highlighted_text = re.sub(pattern, replacement, highlighted_text, flags=re.IGNORECASE)
        
    return highlighted_text


with st.sidebar:
    st.title("📊 Consensus — Quick Info")
    st.markdown("---")

    st.markdown(
        "**📌 About the Project:**\n"
        "Consensus analyzes movie reviews to identify audience sentiment and observe how opinions change across multiple reviews."
    )

    st.markdown(
        "**🧠 Training Data:**\n"
        "The model is trained on the **IMDB 50K Movie Review Dataset**, which contains only positive and negative labels."
    )

    st.markdown(
        "**📊 Sentiment Labels:**\n"
        "- 🟢 **Satisfied:** Confidence above 60%\n"
        "- 🔴 **Disappointed:** Confidence below 40%\n"
        "- 🟡 **Mixed Feelings:** Confidence between 40% and 60%"
    )

    st.markdown("---")

    st.markdown("**⚠️ Known Limitations:**")
    st.markdown(
        "- Mixed sentiment is threshold-based (not directly trained)\n"
        "- Sensitive to spelling mistakes or very short reviews\n"
        "- May struggle with sarcasm or complex expressions"
    )

    st.markdown("---")

    st.link_button("🔗 View Project on GitHub", "https://github.com/satyampandya/Consensus-Sentiment-Volatility-Analysis.git")

st.title("📊 Consensus: Sentiment & Volatility Analysis")
tab1, tab2 = st.tabs(["💬 Single Review & XAI", "📁 Bulk CSV & EDA Analysis"])

# --- TAB 1: SINGLE REVIEW ---
with tab1:
    user_input = st.text_area("Type a movie review here:", placeholder="e.g., The visuals were amazing, but the acting was terrible...")
    
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            label, confidence = sentiment_engine.predict_sentiment(user_input)
            
            st.markdown("### Result:")
            if "Satisfied" in label: st.success(f"**{label}** ({confidence*100:.1f}% Positive)")
            elif "Disappointed" in label: st.error(f"**{label}** ({confidence*100:.1f}% Positive)")
            else: st.warning(f"**{label}** ({confidence*100:.1f}% Positive)")
            
            st.markdown("---")
            st.markdown("### 🧠 Explainable AI (Using LIME)")
            st.caption("Green words pushed the AI toward Positive. Red words pushed it toward Negative.")
            
            with st.spinner("LIME is analyzing model weights..."):
                lime_weights = sentiment_engine.get_lime_explanation(user_input)
                html_text = render_highlighted_text(user_input, lime_weights)
                st.markdown(f'<div style="font-size: 1.2rem; line-height: 1.8; padding: 20px; background-color: #f8f9fa; border-radius: 10px; color: black;">{html_text}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter some text first.")

# --- TAB 2: BULK CSV UPLOAD ---
with tab2:
    uploaded_file = st.file_uploader("Upload a CSV file containing a 'review' column", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = [str(col).lower().strip() for col in df.columns] 
        
        if 'review' not in df.columns:
            st.error("The uploaded CSV must contain a column named 'review'.")
        else:
            with st.spinner("Analyzing reviews & generating metrics..."):
                results = df['review'].apply(lambda x: sentiment_engine.predict_sentiment(str(x)))
                df['Sentiment'] = [res[0] for res in results]
                df['Positive_Score'] = [res[1] for res in results]
                df['Word_Count'] = df['review'].apply(lambda x: len(str(x).split()))
                
                # --- CHARTS & EDA ---
                date_col = next((col for col in df.columns if col in ['date', 'timestamp', 'time', 'review_date']), None)
                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    df = df.sort_values(by=date_col)
                
                col1, col2 = st.columns(2)
                with col1: st.plotly_chart(visualizations.plot_sentiment_pie(df), use_container_width=True)
                with col2: st.plotly_chart(visualizations.plot_sentiment_scatter(df, date_col), use_container_width=True)
                
                st.markdown("### 📊 Exploratory Data Analysis (EDA)")
                st.plotly_chart(visualizations.plot_eda_length_vs_sentiment(df), use_container_width=True)

                # --- DATA PREVIEW & DOWNLOAD ---
                st.markdown("### Processed Data")
                display_df = df.copy()
                st.dataframe(display_df[['review', 'Sentiment', 'Positive_Score', 'Word_Count']].head(50))
                
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Analyzed Data as CSV",
                    data=csv,
                    file_name='consensus_analyzed_reviews.csv',
                    mime='text/csv',
                )