import streamlit as st
import speech_recognition as sr
from sentiment_analysis import analyze_sentiment, transcribe_with_chunks
from product_recommender import ProductRecommender
from objection_handler import ObjectionHandler
from google_sheets import store_data_in_sheet
from sentence_transformers import SentenceTransformer
from env_setup import config

# Initialize the ProductRecommender and ObjectionHandler
product_recommender = ProductRecommender(r"C:\Users\shaik\Downloads\Sales Calls Transcriptions - Sheet2.csv")
objection_handler = ObjectionHandler(r"C:\Users\shaik\Downloads\Sales Calls Transcriptions - Sheet3.csv")
model = SentenceTransformer('all-MiniLM-L6-v2')

def real_time_analysis():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    st.info("Say 'stop' to end the process.")
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

                # Sentiment analysis
                sentiment, score = analyze_sentiment(text)
                st.write(f"*Sentiment:* {sentiment} (Score: {score})")

                # Objection handling
                objection_response = handle_objection(text)
                st.write(f"*Objection Response:* {objection_response}")

                # Product recommendation
                recommendations = product_recommender.get_recommendations(text)
                st.write("*Product Recommendations:*")
                for idx, rec in enumerate(recommendations, 1):
                    st.write(f"{idx}. {rec}")

            except sr.UnknownValueError:
                st.error("Speech Recognition could not understand the audio.")
            except sr.RequestError as e:
                st.error(f"Error with the Speech Recognition service: {e}")
            except Exception as e:
                st.error(f"Error during processing: {e}")

    except Exception as e:
        st.error(f"Error in real-time analysis: {e}")

def handle_objection(text):
    # Generate an objection response based on similarity with known objections
    query_embedding = model.encode([text])
    distances, indices = objection_handler.index.search(query_embedding, 1)
    
    # If similarity is high enough, return objection response
    if distances[0][0] < 1.5:  # Adjust similarity threshold as needed
        responses = objection_handler.handle_objection(text)
        return "\n".join(responses) if responses else "No objection response found."
    return "No objection response found."

def run_app():
    st.title("Real-Time Sales Call Analysis")

    # Start real-time analysis on button click
    if st.button("Start Listening"):
        real_time_analysis()

    # Option to upload sales call transcription (if available)
    uploaded_file = st.file_uploader("Upload Sales Call Transcription", type=["txt", "csv"])
    if uploaded_file is not None:
        st.write("File uploaded successfully.")
        # Process the uploaded file here (parse, analyze, store, etc.)

        # Example of displaying the text content of a file
        if uploaded_file.type == "text/csv":
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            st.write(df)

    # Option to manually input call transcript text for analysis
    st.subheader("Or manually input call transcript for analysis")
    input_text = st.text_area("Paste call transcript here")
    if input_text:
        sentiment, score = analyze_sentiment(input_text)
        st.write(f"Sentiment: {sentiment} (Score: {score})")
        st.write("Product Recommendations:")
        recommendations = product_recommender.get_recommendations(input_text)
        for idx, rec in enumerate(recommendations, 1):
            st.write(f"{idx}. {rec}")
        objection_response = handle_objection(input_text)
        st.write(f"Objection Response: {objection_response}")

if __name__ == "__main__":
    run_app()
