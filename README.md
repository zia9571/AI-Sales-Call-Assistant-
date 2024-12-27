# Real-Time AI Sales Call Assistant for Enhanced Conversation Strategies

## Overview

This project develops a real-time AI-powered assistant designed to enhance sales conversations. The assistant analyzes live sales calls, transcribes speech to text, performs sentiment analysis, and generates insights to optimize conversation strategies. It leverages cutting-edge AI models for speech-to-text processing, sentiment analysis, and conversation summarization.

## Features

- **Real-time Speech-to-Text**: 
    - The system captures audio from a live sales call and transcribes it in real time using **Whisper AI** and **Vosk**.
    
- **Sentiment Analysis**: 
    - Each chunk of conversation is analyzed for sentiment using a multilingual sentiment analysis model (based on **DistilBERT**).
    
- **Conversation Segmentation**: 
    - The audio is divided into chunks based on pauses, allowing for detailed analysis of specific parts of the conversation.
    
- **Conversation Summary**: 
    - At the end of the call, the system generates a summary of the conversation along with the overall sentiment.
    
- **Google Sheets Integration**: 
    - Call data, including chunks, sentiment scores, and summaries, are stored in **Google Sheets** for further analysis.

## Technologies Used

- **Speech-to-Text**: 
    - [Whisper AI](https://huggingface.co/), powered by **Hugging Face**.
    - [Vosk](https://alphacephei.com/vosk/), a speech recognition toolkit for real-time transcription.
  
- **Sentiment Analysis**: 
    - [DistilBERT](https://huggingface.co/distilbert), a transformer-based model fine-tuned for sentiment analysis in multiple languages.

- **Google Sheets Integration**: 
    - Google Sheets API for storing conversation data.
  
- **Other Libraries**:
    - **pyaudio** for capturing audio input.
    - **transformers** from Hugging Face for sentiment analysis and model handling.
    - **dotenv** for loading environment variables securely.
    - **pandas** for data handling.

## Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/ai-sales-call-assistant.git
    cd ai-sales-call-assistant
    ```

2. **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up the environment**:
    - Create a `.env` file in the root directory with the following keys:
      ```
      vosk_model_path=path/to/vosk/model
      huggingface_api_key=your_huggingface_api_key
      google_creds=path/to/your/google/credentials.json
      google_sheet_id=your_google_sheet_id
      ```

5. **Run the project**:
    ```bash
    python main.py
    ```

    This will start the real-time transcription and sentiment analysis.

## Workflow

1. The system listens to a live sales call and transcribes the speech into text using Whisper AI and Vosk.
2. It segments the conversation into chunks based on pauses and performs sentiment analysis on each chunk.
3. The sentiment for each chunk is displayed, and at the end of the call, a summary is generated with an overall sentiment.
4. Data is stored in a Google Sheet, including:
   - **Call ID**
   - **Chunks of Conversation**
   - **Sentiment for each Chunk**
   - **Overall Sentiment**
   - **Conversation Summary**
