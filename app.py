from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

app = Flask(__name__)

# Configuration
folder_path = "db"
cached_llm = Ollama(model="phi3")
embedding = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)
raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)


# Error handlers
@app.errorhandler(Exception)
def handle_error(e):
    app.logger.error(f"Unhandled exception: {str(e)}")
    return jsonify(error="Internal Server Error"), 500


@app.errorhandler(404)
def handle_not_found_error(e):
    return jsonify(error="Not Found"), 404


# API endpoints
@app.route("/ai", methods=["POST"])
def aiPost():
    try:
        json_content = request.json
        query = json_content.get("query")
        response = cached_llm.invoke(query)
        return {"answer": response}
    except Exception as e:
        app.logger.error(f"Error in /ai endpoint: {str(e)}")
        return jsonify(error="An error occurred in AI endpoint"), 500


@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    try:
        json_content = request.json
        query = json_content.get("query")
        
        # Loading vector store
        vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

        # Creating chain
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 20,
                "score_threshold": 0.1,
            },
        )

        document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
        chain = create_retrieval_chain(retriever, document_chain)

        result = chain.invoke({"input": query})

        sources = []
        for doc in result["context"]:
            sources.append(
                {"source": doc.metadata["source"], "page_content": doc.page_content}
            )

        return {"answer": result["answer"], "sources": sources}
    except Exception as e:
        app.logger.error(f"Error in /ask_pdf endpoint: {str(e)}")
        return jsonify(error="An error occurred in ask PDF endpoint"), 500


@app.route("/pdf", methods=["POST"])
def pdfPost():
    try:
        file = request.files["file"]
        file_name = file.filename
        save_file = "pdf/" + file_name
        file.save(save_file)
        
        loader = PDFPlumberLoader(save_file)
        docs = loader.load_and_split()

        chunks = text_splitter.split_documents(docs)

        vector_store = Chroma.from_documents(
            documents=chunks, embedding=embedding, persist_directory=folder_path
        )

        vector_store.persist()

        response = {
            "status": "Successfully Uploaded",
            "filename": file_name,
            "doc_len": len(docs),
            "chunks": len(chunks),
        }
        return response
    except KeyError:
        return jsonify(error="No file part in the request"), 400
    except Exception as e:
        app.logger.error(f"Error in /pdf endpoint: {str(e)}")
        return jsonify(error="An error occurred in PDF endpoint"), 500


def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()
