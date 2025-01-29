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
import streamlit as st

product_titles_df = pd.read_csv(r"C:\Users\shaik\Downloads\Sales Calls Transcriptions - Sheet2.csv")
product_titles = product_titles_df['product_title'].tolist()

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

def filter_product_mentions(chunks, product_titles):
    product_mentions = {}
    for chunk in chunks:
        for product in product_titles:
            if product.lower() in chunk[0].lower():
                if product in product_mentions:
                    product_mentions[product] += 1
                else:
                    product_mentions[product] = 1
    return product_mentions

def run_app():
    st.set_page_config(page_title="Sales Call Assistant", layout="wide")
    st.title("AI Sales Call Assistant")

    st.markdown("""
        <style>
            /* Header Container Styling */
            .header-container {
                background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
                padding: 20px;
                border-radius: 15px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            /* Section Container Styling */
            .section {
                background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
                padding: 25px;
                border-radius: 15px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            /* Header Text Styling */
            .header {
                font-size: 2.5em;
                font-weight: 800;
                text-align: center;
                background: linear-gradient(120deg, #0D6EFD 0%, #0B5ED7 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0;
                padding: 10px;
                letter-spacing: 1px;
            }

            /* Subheader Styling */
            .subheader {
                font-size: 1.8em;
                font-weight: 600;
                background: linear-gradient(120deg, #0D6EFD 0%, #0B5ED7 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-top: 20px;
                margin-bottom: 10px;
                text-align: left;
            }

            /* Table Container Styling */
            .table-container {
                background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }

            /* Dark mode adjustments */
            @media (prefers-color-scheme: dark) {
                .header-container {
                    background: linear-gradient(135deg, #212529 0%, #343A40 100%);
                }
                
                .section {
                    background: linear-gradient(135deg, #212529 0%, #2B3035 100%);
                }
                
                .table-container {
                    background: linear-gradient(135deg, #212529 0%, #2B3035 100%);
                }
                
                .header {
                    background: linear-gradient(120deg, #6EA8FE 0%, #9EC5FE 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }
                
                .subheader {
                    background: linear-gradient(120deg, #6EA8FE 0%, #9EC5FE 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }
            }

            /* Button Styling */
            .stButton > button {
                background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                transition: all 0.3s ease;
            }

            .stButton > button:hover {
                background: linear-gradient(135deg, #1976D2 0%, #1565C0 100%);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            }

            /* Tab Styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 24px;
                background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
                padding: 10px;
                border-radius: 10px;
            }

            .stTabs [data-baseweb="tab"] {
                background-color: transparent;
                border-radius: 4px;
                color: #1976D2;
                font-weight: 600;
                padding: 10px 16px;
            }

            .stTabs [aria-selected="true"] {
                background: linear-gradient(120deg, #2196F3 0%, #1976D2 100%);
                color: white;
            }

            /* Dark mode tab adjustments */
            @media (prefers-color-scheme: dark) {
                .stTabs [data-baseweb="tab-list"] {
                    background: linear-gradient(135deg, #212529 0%, #343A40 100%);
                }
                
                .stTabs [data-baseweb="tab"] {
                    color: #82B1FF;
                }
                
                .stTabs [aria-selected="true"] {
                    background: linear-gradient(120deg, #448AFF 0%, #2979FF 100%);
                }
            }

            /* Message Styling */
            .success {
                background: linear-gradient(135deg, #43A047 0%, #2E7D32 100%);
                color: white;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }

            .error {
                background: linear-gradient(135deg, #E53935 0%, #C62828 100%);
                color: white;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }

            .warning {
                background: linear-gradient(135deg, #FB8C00 0%, #F57C00 100%);
                color: white;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
        </style>
    """, unsafe_allow_html=True)
    


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

                product_mentions = filter_product_mentions(data[['Chunk']].values.tolist(), product_titles)
                product_mentions_df = pd.DataFrame(list(product_mentions.items()), columns=['Product', 'Count'])

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Sentiment Distribution")
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

                with col2:
                    st.subheader("Most Mentioned Products")
                    fig_products = px.pie(
                        values=product_mentions_df['Count'], 
                        names=product_mentions_df['Product'], 
                        title='Most Mentioned Products'
                    )
                    st.plotly_chart(fig_products)

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
                    
                else:
                    st.error("No details available for the selected Call ID.")
        except Exception as e:
            st.error(f"Error loading dashboard: {e}")

if __name__ == "__main__":
    run_app()
