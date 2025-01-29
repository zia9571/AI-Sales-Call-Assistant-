import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import streamlit as st

def generate_insights(call_data):
    """
    Generate ML insights and visualizations from call data
    """
    # Convert call data to DataFrame
    df = pd.DataFrame(call_data)
    
    # Sentiment distribution pie chart
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['sentiment'].value_counts()
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
    plt.title('Sentiment Distribution')
    st.pyplot(plt)
    plt.close()

    # Calculate sentiment trend
    df['sentiment_numeric'] = df['sentiment'].map({'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0})
    
    # Simple trend analysis
    X = np.array(range(len(df))).reshape(-1, 1)
    y = df['sentiment_numeric'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict trend
    trend_score = model.coef_[0]
    trend_interpretation = (
        "Improving" if trend_score > 0.1 else 
        "Declining" if trend_score < -0.1 else 
        "Stable"
    )
    
    # Summary metrics
    st.subheader("Call Analysis Summary")
    st.write(f"Total Calls: {len(df)}")
    st.write("Sentiment Breakdown:")
    st.write(sentiment_counts)
    st.write(f"Sentiment Trend: {trend_interpretation}")

def main():
    st.title("Sales Call Insights")
    
    # Placeholder for loading data mechanism
    st.write("Insights generation ready.")

if __name__ == "__main__":
    main()

