# AI-Sales-Call-Assistant

This project involves creating an AI-driven assistant that analyzes real-time speech during sales calls. The system processes audio input from the microphone, transcribes it to text, performs sentiment analysis on each conversation chunk, and provides a summary of the entire call. The data is then stored in a Google Sheet, enabling easy review and analysis of the conversation's tone and content.

Features
Real-time Speech-to-Text: The system captures audio from a live sales call and transcribes it in real time using Whisper AI and Vosk.
Sentiment Analysis: Each chunk of conversation is analyzed for sentiment using a multilingual sentiment analysis model (based on DistilBERT).
Conversation Segmentation: The audio is divided into chunks based on pauses, allowing for detailed analysis of specific parts of the conversation.
Conversation Summary: At the end of the call, the system generates a summary of the conversation along with the overall sentiment.
Google Sheets Integration: Call data, including chunks, sentiment scores, and summaries, are stored in Google Sheets for further analysis.
