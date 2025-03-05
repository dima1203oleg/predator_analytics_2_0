import os
import json
import requests
from flask import Flask, request, jsonify
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import (
    UnstructuredPDFLoader, UnstructuredImageLoader, UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from redis import Redis
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncpg

app = Flask(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OPENSEARCH_HOSTS = os.getenv("OPENSEARCH_HOSTS", "http://localhost:9200").split(",")

redis_client = Redis(host=REDIS_HOST, port=6379, decode_responses=True)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_db_pool():
    return await asyncpg.create_pool(
        database="predator_db",
        user="predator",
        password="strong_password",
        host=POSTGRES_HOST,
        command_timeout=10
    )

embeddings = OllamaEmbeddings(model="mistral", base_url=OLLAMA_HOST)
llm = Ollama(model="mistral", base_url=OLLAMA_HOST)

vectorstore = OpenSearchVectorSearch(
    index_name="predator_index",
    opensearch_url=OPENSEARCH_HOSTS[0],
    embedding_function=embeddings,
    http_auth=("admin", "strong_password"),
    use_ssl=False,
    bulk_size=1000
)

RAG_PROMPT_TEMPLATE = """
Ти аналітик митних схем та фінансових махінацій. Проаналізуй наступні дані та відповідай на питання.
Дані:
{context}
Питання: {question}
Відповідь:
"""
prompt = PromptTemplate(template=RAG_PROMPT_TEMPLATE, input_variables=["context", "question"])

@app.route('/api/version', methods=['GET'])
def get_version():
    return jsonify({"version": "1.0"}), 200

@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        ollama_tags_url = f"{OLLAMA_HOST}/api/tags"
        response = requests.get(ollama_tags_url, timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return jsonify([{"id": m["name"], "name": m["name"]} for m in models]), 200
        return jsonify([{"id": "mistral", "name": "Mistral Model"}]), 200
    except Exception:
        return jsonify([{"id": "mistral", "name": "Mistral Model"}]), 200

@app.route('/api/tags', methods=['GET'])
def get_tags():
    tags = [{"id": "mistral", "name": "Mistral Model"}, {"id": "custom", "name": "Custom Analytics Model"}]
    return jsonify(tags), 200

@app.route('/process_query', methods=['POST'])
async def process_query():
    try:
        query = request.json.get("query")
        if not query:
            return jsonify({"error": "Query is required"}), 400

        cache_key = f"query_cache:{query}"
        if cached := redis_client.get(cache_key):
            return jsonify({"source": "cache", "response": json.loads(cached)})

        query_embedding = embeddings.embed_query(query)
        docs = vectorstore.similarity_search_by_vector(query_embedding, k=5)

        pool = await get_db_pool()
        async with pool.acquire() as conn:
            sql_results = await conn.fetch("""
                SELECT "Номер МД", "Опис товару", "Відправник", "Одержувач"
                FROM customs_data
                WHERE "Опис товару" ILIKE $1 OR "Відправник" ILIKE $1 OR "Одержувач" ILIKE $1
                LIMIT 5
            """, f"%{query}%")

        context = "\n".join(doc.page_content for doc in docs) + "\nSQL Results:\n" + "\n".join(str(row) for row in sql_results)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        response = qa_chain({"query": query, "context": context})

        final_response = {"answer": response["result"], "sources": [doc.metadata for doc in docs] + [{"sql": dict(row)} for row in sql_results]}
        redis_client.setex(cache_key, 3600, json.dumps(final_response))

        await log_query(query, final_response["answer"])
        return jsonify({"source": "rag", "response": final_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_document', methods=['POST'])
async def upload_document():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Файл не надано"}), 400

        file = request.files['file']
        filename = file.filename.lower()
        temp_path = f"/tmp/{filename}"
        file.save(temp_path)

        if filename.endswith(".pdf"):
            loader = UnstructuredPDFLoader(temp_path)
        elif filename.endswith((".jpeg", ".jpg", ".png")):
            loader = UnstructuredImageLoader(temp_path)
        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(temp_path)
        else:
            return jsonify({"error": "Непідтримуваний формат файлу"}), 400

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)
        vectorstore.add_texts([doc.page_content for doc in split_docs], metadatas=[{"filename": filename} for _ in split_docs])

        return jsonify({"status": "ok", "message": f"Файл {filename} успішно оброблено."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

async def log_query(query, answer):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("INSERT INTO query_log (query, response) VALUES ($1, $2)", query, answer)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})