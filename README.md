# Real-Time AI Sales Call Assistant for Enhanced Conversation

## Overview
![Logo](data/logo.jpg)

This project develops a real-time AI-powered assistant designed to enhance sales conversations. The assistant analyzes live sales calls, transcribes speech to text, performs sentiment analysis, and generates product recommendations and objection handling responses.

## Goals & Objectives

- **Improve Sales Performance:** Provide real-time feedback and actionable insights during sales calls.
- **Enhance Customer Experience:** Enable personalized interactions with customers.
- **Monitor Sales Performance:** Allow sales managers to monitor and analyze sales performance.

## Target Users

- **Sales Representatives:** Benefit from real-time feedback, product recommendations, and objection handling during sales calls.
- **Sales Managers:** Gain insights into team performance and customer interactions through comprehensive call summaries and sentiment analysis.

## Technology Stack

- **Languages:** Python
- **Frameworks:** Streamlit
- **APIs:** Google Sheets API, Hugging Face API
- **Libraries:** SpeechRecognition, Vosk, Sentence Transformers, Faiss, Transformers, Pandas, Plotly, PyAudio
- **Databases:** Google Sheets
- **Other Tools:** dotenv for environment variable management

## Features

- **Real-Time Speech-to-Text:** Captures and transcribes audio from live sales calls using Vosk.
- **Sentiment Analysis:** Analyzes the sentiment of conversation chunks using DistilBERT.
- **Product Recommendations:** Provides relevant product recommendations based on the conversation.
- **Objection Handling:** Offers responses to customer objections during sales calls.
- **Dashboard:** Visualizes call summaries and sentiment analysis using Streamlit.
- **Data Storage:** Stores call data, including sentiment scores and summaries, in Google Sheets.

## Setup Instructions

1. **Clone the repository:**
    ```bash
    git clone https://github.com/zia9571/AI-Sales-Call-Assistant.git
    cd AI-Sales-Call-Assistant
    ```

2. **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up the environment:**
   Create a `.env` file in the root directory with the following keys:
      ```
      vosk_model_path=path/to/vosk/model
      huggingface_api_key=your_huggingface_api_key
      google_creds=path/to/your/google/credentials.json
      google_sheet_id=your_google_sheet_id
      ```

5. **Run the project:**
    ```bash
    streamlit run app.py
    ```

This will start the real-time transcription and sentiment analysis.

## Workflow

1. The system listens to a live sales call and transcribes the speech into text using Vosk.
2. It segments the conversation into chunks based on pauses and performs sentiment analysis on each chunk.
3. The sentiment for each chunk is displayed, and at the end of the call, a summary is generated with an overall sentiment.
4. Data is stored in a Google Sheet, including:
   - **Call ID**
   - **Chunks of Conversation**
   - **Sentiment for each Chunk**
   - **Overall Sentiment**
   - **Conversation Summary**

## Model Architecture

The Real-Time AI Sales Call Assistant leverages a combination of speech recognition, sentiment analysis, and machine learning models to provide real-time insights and recommendations. The architecture includes:
- **Speech Recognition:** Vosk for real-time audio transcription.
- **Sentiment Analysis:** DistilBERT model from Hugging Face for analyzing conversation sentiment.
- **Product Recommendations:** Sentence Transformers and Faiss for recommending relevant products.
- **Objection Handling:** Sentence Transformers and Faiss for providing responses to customer objections.
- **Data Storage:** Google Sheets for storing and retrieving call data.
- **Dashboard:** Streamlit for visualizing call summaries and sentiment analysis.

## Insights and Learning

Adopting a sailboat perspective, the project journey can be summarized as follows:

- **Wind (Favorable Factors):**
  - Integration of advanced technologies like Vosk and DistilBERT.
  - Real-time processing providing immediate feedback.
  - User-friendly dashboard for easy monitoring and analysis.

- **Anchors (Challenges):**
  - Ensuring accurate speech recognition in noisy environments.
  - Balancing model performance and accuracy.
  - Managing dependencies and environment setup.

- **Rocks (Potential Risks Avoided):**
  - Secure data storage and access with Google Sheets.
  - Avoiding model overfitting through regular validation and testing.

- **Personal Learnings:**
  - Importance of real-time feedback in improving sales performance.
  - Continuous improvement and fine-tuning of models.
  - Collaboration and integration of various system components.

## Future Work

- Enhance model accuracy and performance.
- Expand multilingual support.
- Develop advanced analytics and reporting features.
- Integrate with CRM systems for seamless data management.

## References

- Vosk: https://alphacephei.com/vosk/
- Hugging Face Transformers: https://huggingface.co/
- Streamlit: https://streamlit.io/
- Google Sheets API: https://developers.google.com/sheets/api
- DistilBERT: https://huggingface.co/distilbert
- Sentence Transformers: https://www.sbert.net/
- Faiss: https://github.com/facebookresearch/faiss
