# Document Search and AI Backend API

This project implements a backend API using Flask for document search and AI-based question answering. It provides endpoints for uploading PDF documents, querying a language model, and retrieving relevant information from stored documents.

## Features
- **Document Upload:** Upload PDF documents to be processed and stored for search operations.
- **AI Query:** Query an AI model to get answers based on provided context and input.
- **Document Retrieval:** Retrieve relevant documents based on similarity scores and user queries.

## Technologies Used
- **Flask:** Web framework used to build the API endpoints and handle HTTP requests.
- **Python Libraries:** Including `langchain_community`, `langchain_text_splitters`, `langchain`, and others for AI models, document handling, and text processing.
- **Chroma Vector Store:** Used for storing document embeddings and facilitating fast retrieval.
