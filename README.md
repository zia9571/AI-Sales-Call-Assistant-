# AI-Sales-Call-Assistant-
Overview
This project involves creating an AI-driven assistant that analyzes real-time speech during sales calls. The system processes audio input from the microphone, transcribes it to text, performs sentiment analysis on each conversation chunk, and provides a summary of the entire call. The data is then stored in a Google Sheet, enabling easy review and analysis of the conversation's tone and content.

Features
*Real-Time Speech-to-Text: Converts live speech during sales calls into text.
-Sentiment Analysis: Detects the sentiment (Positive, Negative, or Neutral) for each conversation chunk using a Hugging Face model.
Conversation Chunking: The speech is segmented into chunks based on pauses (3 seconds or more).
Conversation Summary: Summarizes the entire conversation after the call ends.
Google Sheets Integration: Stores conversation chunks, sentiment analysis results, and a summary in a Google Sheet for easy tracking and review.

