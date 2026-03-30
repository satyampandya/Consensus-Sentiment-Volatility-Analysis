import plotly.express as px

def plot_sentiment_pie(df):
    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    return px.pie(
        sentiment_counts, names='Sentiment', values='Count', title='Overall Audience Sentiment',
        color='Sentiment',
        color_discrete_map={"🟢 Satisfied": "#28a745", "🔴 Disappointed": "#dc3545", "🟡 Mixed Feelings": "#ffc107"}
    )

def plot_sentiment_scatter(df, date_col=None):
    if date_col:
        x_axis, x_label, title = date_col, "Timeline (Actual Date)", 'Sentiment Distribution Over Time'
    else:
        df['Index'] = range(len(df))
        x_axis, x_label, title = 'Sentiment Distribution (Sequential Order)'

    fig = px.scatter(
        df, x=x_axis, y='Positive_Score', color='Sentiment', title=title,
        labels={x_axis: x_label, 'Positive_Score': 'AI Positive Confidence'},
        color_discrete_map={"🟢 Satisfied": "#28a745", "🔴 Disappointed": "#dc3545", "🟡 Mixed Feelings": "#ffc107"}
    )
    
    fig.add_hline(y=0.60, line_dash="dash", line_color="green")
    fig.add_hline(y=0.40, line_dash="dash", line_color="red")
    return fig

def plot_eda_length_vs_sentiment(df):
    fig = px.box(
        df, x="Sentiment", y="Word_Count", color="Sentiment",
        title="EDA: Review Length vs. Sentiment",
        labels={"Word_Count": "Number of Words in Review"},
        color_discrete_map={"🟢 Satisfied": "#28a745", "🔴 Disappointed": "#dc3545", "🟡 Mixed Feelings": "#ffc107"}
    )
    return fig