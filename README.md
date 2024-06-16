# Document Search and AI Backend API

This project implements a backend API using Flask for document search and AI-based question answering. It provides endpoints for uploading PDF documents, querying a language model, and retrieving relevant information from stored documents.

## Features

- **Document Upload**: Upload PDF documents to be processed and stored for search operations.
- **AI Query**: Query an AI model to get answers based on provided context and input.
- **Document Retrieval**: Retrieve relevant documents based on similarity scores and user queries.

## Technologies Used

- **Flask**: Web framework used to build the API endpoints and handle HTTP requests.
- **Python Libraries**:
  - langchain_community
  - langchain_text_splitters
  - langchain
  - PDFPlumberLoader
- **Chroma Vector Store**: Used for storing document embeddings and facilitating fast retrieval.

## API Endpoints

- **POST /ai**: Endpoint to query an AI model for answering questions.
- **POST /ask_pdf**: Endpoint to retrieve relevant documents based on user queries.
- **POST /pdf**: Endpoint to upload PDF documents for processing and storage.

## Configuration

- **folder_path**: Path to the directory where documents are stored (`db` by default).
- **cached_llm**: Instance of Ollama language model (`phi3` model).
- **embedding**: FastEmbedEmbeddings used for text embeddings.
- **text_splitter**: RecursiveCharacterTextSplitter configuration for text segmentation.
- **raw_prompt**: Template for AI query responses.

## Error Handling

The API includes error handling for common HTTP errors and internal server errors. All exceptions are logged with detailed error messages.

## Setup and Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
