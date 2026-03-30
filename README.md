# 📊 Consensus — Sentiment & Volatility Analysis

Consensus is a machine learning–based sentiment analysis system designed to understand audience opinion from movie reviews. It classifies reviews as **Satisfied**, **Disappointed**, or **Mixed**, and visualizes how sentiment changes across multiple reviews over time using an interactive Streamlit interface.

🎯 Try it Live: https://consensus-ai.streamlit.app/

---

# 🚀 Features

- ✅ Predict sentiment from a single movie review
- 📁 Upload CSV files for bulk sentiment analysis
- 📊 Sentiment distribution visualization (pie chart)
- 📈 Sentiment change over time (volatility analysis)
- 🧠 Explainable AI using LIME word highlighting
- 📥 Download analyzed dataset as CSV

---

# 🧠 Technologies Used

- Frontend: Streamlit
- Machine Learning: Scikit-learn, LinearSVC, TF-IDF
- Visualization: Plotly
- Explainability: LIME
- Data Processing: Pandas, NumPy
- Scraping: Selenium, BeautifulSoup

---

# Data Used

- [IMDB 50k Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Scraped data attached with scraper script

# 📂 Project Structure

```
Consensus/
├── dashboard/
│   ├── app.py
│   └── visualizations.py
├── src/
│   ├── sentiment_engine.py
│   └── train_model.py
├── models/
├── datasets/
├── test_datasets/
├── requirements.txt
└── README.md
```

---

# 🧪 How to Run Locally

Clone the repository

```
git clone https://github.com/satyampandya/Consensus-Sentiment-Volatility-Analysis
cd Consensus
```

Install requirements

```
pip install -r requirements.txt
```

Run the application

```
streamlit run dashboard/app.py
```
---

# ⚠️ Limitations

| Engine           | Limitation                                                   |
| ---------------- | ------------------------------------------------------------ |
| ML Model         | Trained on binary sentiment dataset (positive/negative)      |
| Mixed Detection  | Uses 40–60% confidence threshold, may not always be accurate |
| Text Quality     | Typing mistakes can affect prediction                        |
| Language Context | May struggle with sarcasm or very short reviews              |
---

# 📬 License

This project is free for learning and research purposes.
Please do not misuse the model or system for misinformation.