import streamlit as st
import speech_recognition as sr
from sentiment_analysis import analyze_sentiment
from product_recommender import ProductRecommender
from objection_handler import ObjectionHandler
from google_sheets import store_data_in_sheet
from sentence_transformers import SentenceTransformer
from env_setup import config
import re

# Initialize the ProductRecommender and ObjectionHandler
product_recommender = ProductRecommender(r"C:\Users\shaik\Downloads\Sales Calls Transcriptions - Sheet2.csv")
objection_handler = ObjectionHandler(r"C:\Users\shaik\Downloads\Sales Calls Transcriptions - Sheet3.csv")
model = SentenceTransformer('all-MiniLM-L6-v2')

def is_valid_input(text):
    text = text.strip().lower()
    if len(text) < 3 or re.match(r'^[a-zA-Z\s]*$', text) is None:
        return False
    return True

def is_relevant_sentiment(sentiment_score):
    return sentiment_score > 0.4  # Adjust this threshold based on your needs

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

                # Append to the total conversation
                total_text += text + " "
                sentiment, score = analyze_sentiment(text)
                sentiment_scores.append(score)
                transcribed_chunks.append((text, sentiment, score))

                st.write(f"*Sentiment:* {sentiment} (Score: {score})")

                # Handle objection if relevant
                objection_response = handle_objection(text)
                st.write(f"*Objection Response:* {objection_response}")

                # Perform product recommendation if input is valid and sentiment is relevant
                if is_valid_input(text) and is_relevant_sentiment(score):
                    query_embedding = model.encode([text])
                    distances, indices = product_recommender.index.search(query_embedding, 1)

                    if distances[0][0] < 1.5:  # Similarity threshold
                        recommendations = product_recommender.get_recommendations(text)
                        st.write("*Product Recommendations:*")
                        for idx, rec in enumerate(recommendations, 1):
                            st.write(f"{idx}. {rec}")
                    else:
                        st.write("No relevant product recommendations based on this input.")
                else:
                    st.write("No product recommendations available due to irrelevant input or negative sentiment.")

            except sr.UnknownValueError:
                st.error("Speech Recognition could not understand the audio.")
            except sr.RequestError as e:
                st.error(f"Error with the Speech Recognition service: {e}")
            except Exception as e:
                st.error(f"Error during processing: {e}")

        # After conversation ends, calculate and display overall sentiment and summary
        overall_sentiment = calculate_overall_sentiment(sentiment_scores)
        st.subheader("Conversation Summary:")
        st.write(total_text.strip())
        st.subheader("Overall Sentiment:")
        st.write(overall_sentiment)

        # Store data in Google Sheets
        store_data_in_sheet(config["google_sheet_id"], transcribed_chunks, total_text.strip(), overall_sentiment)
        st.success("Conversation data stored successfully in Google Sheets!")

    except Exception as e:
        st.error(f"Error in real-time analysis: {e}")

def handle_objection(text):
    query_embedding = model.encode([text])
    distances, indices = objection_handler.index.search(query_embedding, 1)
    if distances[0][0] < 1.5:  # Adjust similarity threshold as needed
        responses = objection_handler.handle_objection(text)
        return "\n".join(responses) if responses else "No objection response found."
    return "No objection response found."

def run_app():
    st.title("Real-Time Sales Call Analysis")

    if st.button("Start Listening"):
        real_time_analysis()

    uploaded_file = st.file_uploader("Upload Sales Call Transcription", type=["txt", "csv"])
    if uploaded_file is not None:
        st.write("File uploaded successfully.")
        if uploaded_file.type == "text/csv":
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            st.write(df)

    st.subheader("Or manually input call transcript for analysis")
    input_text = st.text_area("Paste call transcript here")
    if input_text:
        sentiment, score = analyze_sentiment(input_text)
        st.write(f"Sentiment: {sentiment} (Score: {score})")

        if is_valid_input(input_text) and is_relevant_sentiment(score):
            st.write("Product Recommendations:")
            query_embedding = model.encode([input_text])
            distances, indices = product_recommender.index.search(query_embedding, 1)
            if distances[0][0] < 1.5:
                recommendations = product_recommender.get_recommendations(input_text)
                for idx, rec in enumerate(recommendations, 1):
                    st.write(f"{idx}. {rec}")
            else:
                st.write("No relevant product recommendations based on this input.")
        
        objection_response = handle_objection(input_text)
        st.write(f"Objection Response: {objection_response}")

if __name__ == "__main__":
    run_app()
