from flask import Flask, request, jsonify
from langchain_community.llms import Ollama

app = Flask(__name__)


cached_llm = Ollama(model="phi3")



@app.route("/ai", methods=["POST"])
def ai_post():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    response = cached_llm.invoke(query)

    print(response)
    
    response_answer = {"answer" : response}
    return response_answer

@app.route("/pdf",methods=["POST"])
def pdf_post():
    print("Post /pdf called")
    file =request.files["file"]
    file_name=file.filename
    savefile="pdf/"+ file_name
    file.save(savefile)
    print(f"filename: {file_name}")
    response={"status" : "successfully Uploaded" , "filename" :file_name}
    return response

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()
