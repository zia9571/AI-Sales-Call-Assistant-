import speech_recognition as sr
from sentiment_analysis import analyze_sentiment
from product_recommender import ProductRecommender
from objection_handler import ObjectionHandler
from google_sheets import fetch_call_data, store_data_in_sheet
from sentence_transformers import SentenceTransformer
from env_setup import config
import re
import uuid
from google.oauth2 import service_account
from googleapiclient.discovery import build
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st 

product_recommender = ProductRecommender(r"C:\Users\shaik\Downloads\Sales Calls Transcriptions - Sheet2.csv")
objection_handler = ObjectionHandler(r"C:\Users\shaik\Downloads\Sales Calls Transcriptions - Sheet3.csv")
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_comprehensive_summary(chunks):
    full_text = " ".join([chunk[0] for chunk in chunks])
    
    total_chunks = len(chunks)
    sentiments = [chunk[1] for chunk in chunks]
    
    context_keywords = {
        'product_inquiry': ['dress', 'product', 'price', 'stock'],
        'pricing': ['cost', 'price', 'budget'],
        'negotiation': ['installment', 'payment', 'manage']
    }
    
    themes = []
    for keyword_type, keywords in context_keywords.items():
        if any(keyword.lower() in full_text.lower() for keyword in keywords):
            themes.append(keyword_type)
    
    positive_count = sentiments.count('POSITIVE')
    negative_count = sentiments.count('NEGATIVE')
    neutral_count = sentiments.count('NEUTRAL')
    
    key_interactions = []
    for chunk in chunks:
        if any(keyword.lower() in chunk[0].lower() for keyword in ['price', 'dress', 'stock', 'installment']):
            key_interactions.append(chunk[0])
    
    summary = f"Conversation Summary:\n"
    
    if 'product_inquiry' in themes:
        summary += "• Customer initiated a product inquiry about items.\n"
    
    if 'pricing' in themes:
        summary += "• Price and budget considerations were discussed.\n"
    
    if 'negotiation' in themes:
        summary += "• Customer and seller explored flexible payment options.\n"
    
    summary += f"\nConversation Sentiment:\n"
    summary += f"• Positive Interactions: {positive_count}\n"
    summary += f"• Negative Interactions: {negative_count}\n"
    summary += f"• Neutral Interactions: {neutral_count}\n"
    
    summary += "\nKey Conversation Points:\n"
    for interaction in key_interactions[:3]:  # Limit to top 3 key points
        summary += f"• {interaction}\n"
    
    if positive_count > negative_count:
        summary += "\nOutcome: Constructive and potentially successful interaction."
    elif negative_count > positive_count:
        summary += "\nOutcome: Interaction may require further follow-up."
    else:
        summary += "\nOutcome: Neutral interaction with potential for future engagement."
    
    return summary

def is_valid_input(text):
    text = text.strip().lower()
    if len(text) < 3 or re.match(r'^[a-zA-Z\s]*$', text) is None:
        return False
    return True

def is_relevant_sentiment(sentiment_score):
    return sentiment_score > 0.4

def calculate_overall_sentiment(sentiment_scores):
    if sentiment_scores:
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        overall_sentiment = (
            "POSITIVE" if average_sentiment > 0 else
            "NEGATIVE" if average_sentiment < 0 else
            "NEUTRAL"
        )
    else:
        overall_sentiment = "NEUTRAL"
    return overall_sentiment

def real_time_analysis():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    st.info("Say 'stop' to end the process.")

    sentiment_scores = []
    transcribed_chunks = []
    total_text = ""

    try:
        while True:
            with mic as source:
                st.write("Listening...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)

            try:
                st.write("Recognizing...")
                text = recognizer.recognize_google(audio)
                st.write(f"*Recognized Text:* {text}")

                if 'stop' in text.lower():
                    st.write("Stopping real-time analysis...")
                    break

                total_text += text + " "
                sentiment, score = analyze_sentiment(text)
                sentiment_scores.append(score)
                
                objection_response = handle_objection(text)

                recommendations = []
                if is_valid_input(text) and is_relevant_sentiment(score):
                    query_embedding = model.encode([text])
                    distances, indices = product_recommender.index.search(query_embedding, 1)

                    if distances[0][0] < 1.5:  
                        recommendations = product_recommender.get_recommendations(text)

                transcribed_chunks.append((text, sentiment, score))

                st.write(f"*Sentiment:* {sentiment} (Score: {score})")
                st.write(f"*Objection Response:* {objection_response}")
                
                if recommendations:
                    st.write("*Product Recommendations:*")
                    for rec in recommendations:
                        st.write(rec)

            except sr.UnknownValueError:
                st.error("Speech Recognition could not understand the audio.")
            except sr.RequestError as e:
                st.error(f"Error with the Speech Recognition service: {e}")
            except Exception as e:
                st.error(f"Error during processing: {e}")

        overall_sentiment = calculate_overall_sentiment(sentiment_scores)
        call_summary = generate_comprehensive_summary(transcribed_chunks)
        
        st.subheader("Conversation Summary:")
        st.write(total_text.strip())
        st.subheader("Overall Sentiment:")
        st.write(overall_sentiment)

        store_data_in_sheet(
            config["google_sheet_id"], 
            transcribed_chunks, 
            call_summary, 
            overall_sentiment
        )
        st.success("Conversation data stored successfully in Google Sheets!")

    except Exception as e:
        st.error(f"Error in real-time analysis: {e}")

def handle_objection(text):
    query_embedding = model.encode([text])
    distances, indices = objection_handler.index.search(query_embedding, 1)
    if distances[0][0] < 1.5: 
        responses = objection_handler.handle_objection(text)
        return "\n".join(responses) if responses else "No objection response found."
    return "No objection response found."


def run_app():
    st.set_page_config(page_title="Sales Call Assistant", layout="wide")
    st.title("AI Sales Call Assistant")

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a mode:", ["Real-Time Call Analysis", "Dashboard"])

    if app_mode == "Real-Time Call Analysis":
        st.header("Real-Time Sales Call Analysis")
        if st.button("Start Listening"):
            real_time_analysis()

    elif app_mode == "Dashboard":
        st.header("Call Summaries and Sentiment Analysis")
        try:
            data = fetch_call_data(config["google_sheet_id"])
            if data.empty:
                st.warning("No data available in the Google Sheet.")
            else:
                sentiment_counts = data['Sentiment'].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Sentiment Distribution")
                    fig_pie = px.pie(
                        values=sentiment_counts.values, 
                        names=sentiment_counts.index, 
                        title='Call Sentiment Breakdown',
                        color_discrete_map={
                            'POSITIVE': 'green', 
                            'NEGATIVE': 'red', 
                            'NEUTRAL': 'blue'
                        }
                    )
                    st.plotly_chart(fig_pie)

                with col2:
                    st.subheader("Sentiment Counts")
                    fig_bar = px.bar(
                        x=sentiment_counts.index, 
                        y=sentiment_counts.values, 
                        title='Number of Calls by Sentiment',
                        labels={'x': 'Sentiment', 'y': 'Number of Calls'},
                        color=sentiment_counts.index,
                        color_discrete_map={
                            'POSITIVE': 'green', 
                            'NEGATIVE': 'red', 
                            'NEUTRAL': 'blue'
                        }
                    )
                    st.plotly_chart(fig_bar)

                st.subheader("All Calls")
                display_data = data.copy()
                display_data['Summary Preview'] = display_data['Summary'].str[:100] + '...'
                st.dataframe(display_data[['Call ID', 'Chunk', 'Sentiment', 'Summary Preview', 'Overall Sentiment']])

                unique_call_ids = data[data['Call ID'] != '']['Call ID'].unique()
                call_id = st.selectbox("Select a Call ID to view details:", unique_call_ids)

                call_details = data[data['Call ID'] == call_id]
                if not call_details.empty:
                    st.subheader("Detailed Call Information")
                    st.write(f"**Call ID:** {call_id}")
                    st.write(f"**Overall Sentiment:** {call_details.iloc[0]['Overall Sentiment']}")
                    
                    st.subheader("Full Call Summary")
                    st.text_area("Summary:", 
                                 value=call_details.iloc[0]['Summary'], 
                                 height=200, 
                                 disabled=True)
                    
                    st.subheader("Conversation Chunks")
                    for _, row in call_details.iterrows():
                        if pd.notna(row['Chunk']):  
                            st.write(f"**Chunk:** {row['Chunk']}")
                            st.write(f"**Sentiment:** {row['Sentiment']}")
                            st.write("---") 
                else:
                    st.error("No details available for the selected Call ID.")
        except Exception as e:
            st.error(f"Error loading dashboard: {e}")

if __name__ == "__main__":
    run_app()
