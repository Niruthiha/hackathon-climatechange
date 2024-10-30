# hackathon-climatechange
Climate Resiliency Hack - Northeastern University 

# ClimateConnect 🌍

An AI-powered climate intelligence platform providing instant, verified climate information.

## Features
- Real-time climate Q&A
- Verified information from trusted sources
- Vector-based semantic search
- Interactive web interface

## Tech Stack
- Python 3.9+
- ChromaDB for vector storage
- Streamlit for web interface
- OpenAI API for LLM capabilities

## Overview
The Climate Change Chatbot is an interactive tool designed to provide users with information about climate change using advanced natural language processing techniques. By leveraging a Retrieval-Augmented Generation (RAG) approach, this chatbot utilizes a combination of a vector database and OpenAI's language model to deliver accurate and relevant responses based on user queries.

## Features
- Natural Language Understanding: The chatbot understands user queries about climate change and retrieves relevant information from a preprocessed dataset.
- Dynamic Responses: Utilizing the GPT-3.5-turbo model, the chatbot generates context-aware responses based on the queried information.
- Easy to Use: Users can simply input their questions into a text field to receive informative answers.

## Files overview
1. app.py
Purpose: This file typically serves as the main entry point for your application. It often handles the web server setup and routes for a web-based application.

2. rag.py
Purpose: This file defines the Retrieval-Augmented Generation (RAG) function used to generate answers based on the provided queries and retrieved documents.

## Technologies Used
- Required Python packages listed in requirements.txt

## Data Source
The data used for the Climate Change Chatbot is derived from the Environment and Climate Change Canada (ECCC)
IPCC (Intergovernmental Panel on Climate Change) Reports, including the latest 2023 Synthesis Report
University of Manitoba climate research and expertise. This technical reports and documents providing essential insights into climate change impacts and responses.

## Evaluating the Model
The script evaluates the fine-tuned model based on several metrics. It uses ChromaDB for document retrieval and the SentenceTransformers library for embedding calculations. Ensure you have the correct path to your ChromaDB storage and adjust the model ID as necessary.
Here’s a concise summary of the model metrics:

Relevance: Measures how well the retrieved documents match the query using cosine similarity between query and document embeddings. Higher scores indicate better relevance.

Correctness: Assesses the accuracy of the generated answer compared to a reference answer using BLEU and ROUGE scores. Higher scores reflect better alignment with the reference.

Faithfulness: Evaluates how accurately the generated answer reflects the information from the retrieved documents, typically rated on a scale from 1 to 5. Ensures the answer is true to the context.

Robustness: Tests the model's consistency by generating responses for various rephrasings of the same query and measuring similarity among those responses. Higher similarity indicates better robustness.

These metrics help gauge how effectively your RAG model retrieves relevant information, generates accurate responses, adheres to context, and maintains performance across different inputs.

## Setup
1. Clone the Repository
2. Set Environment Variable: Make sure to set your OpenAI API key in your environment: export OPENAI_API_KEY="your_api_key_here"
3. Running the Chatbot: To run the chatbot, execute the following command: streamlit run app.py

## Create a Virtual Environment
It’s recommended to use a virtual environment to manage project dependencies. 

## Install Required Libraries
Once the virtual environment is activated, install the required dependencies using the requirements.txt file. 
This will install all necessary libraries, including:
- openai: For accessing OpenAI's API.
- langchain: For text splitting and management.
- chromadb: For creating and querying the vector database.
- pypdf: For reading PDF documents.

