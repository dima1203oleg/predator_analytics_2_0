import os
import json
import requests
import sys
import traceback
import csv
import io
from flask import Flask, request, jsonify, send_file
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    UnstructuredPDFLoader, UnstructuredImageLoader, UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader, CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from redis import Redis
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncpg
import logging
from openpyxl import load_workbook  # Додаємо openpyxl для обробки великих Excel-файлів

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OPENSEARCH_HOSTS = os.getenv("OPENSEARCH_HOSTS", "http://localhost:9200").split(",")
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", 2048))
OPENSEARCH_BULK_SIZE = int(os.getenv("OPENSEARCH_BULK_SIZE", 500))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10 MB за замовчуванням

logger.info(f"Connecting to Redis: {REDIS_HOST}")
redis_client = Redis(host=REDIS_HOST, port=6379, decode_responses=True)

logger.info(f"Initializing embeddings with Ollama: {OLLAMA_HOST}")
embeddings = OllamaEmbeddings(model="mistral", base_url=OLLAMA_HOST)
llm = Ollama(model="mistral", base_url=OLLAMA_HOST)

logger.info(f"Connecting to OpenSearch: {OPENSEARCH_HOSTS[0]}")
vectorstore = OpenSearchVectorSearch(
    index_name="customs_data",
    opensearch_url=OPENSEARCH_HOSTS[0],
    embedding_function=embeddings,
    http_auth=("admin", "strong_password"),
    use_ssl=False,
    bulk_size=OPENSEARCH_BULK_SIZE
)

RAG_PROMPT_TEMPLATE = """
You are a customs schemes and financial fraud analyst. Analyze the following data and answer the question.
Data:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=RAG_PROMPT_TEMPLATE, input_variables=["context", "question"])

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_db_pool():
    logger.info(f"Connecting to PostgreSQL: {POSTGRES_HOST}")
    return await asyncpg.create_pool(
        database="predator_db",
        user="predator",
        password="strong_password",
        host=POSTGRES_HOST,
        command_timeout=10
    )

@app.route('/api/version', methods=['GET'])
def get_version():
    try:
        ollama_version_url = f"{OLLAMA_HOST}/api/version"
        response = requests.get(ollama_version_url, timeout=5)
        if response.status_code == 200:
            ollama_version = response.json().get("version", "unknown")
            return jsonify({"version": ollama_version}), 200
        else:
            logger.warning(f"Failed to fetch version from Ollama, status: {response.status_code}")
    except Exception as e:
        logger.error(f"Error fetching version from Ollama: {e}")
    return jsonify({"version": "0.5.13"}), 200

@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        ollama_tags_url = f"{OLLAMA_HOST}/api/tags"
        response = requests.get(ollama_tags_url, timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return jsonify([{"id": m["name"], "name": m["name"]} for m in models]), 200
        return jsonify([{"id": "mistral", "name": "Mistral Model"}]), 200
    except Exception as e:
        logger.error(f"Error fetching models from Ollama: {e}")
        return jsonify([{"id": "mistral", "name": "Mistral Model"}]), 200

@app.route('/api/tags', methods=['GET'])
def get_tags():
    tags = [
        {
            "name": "mistral:latest",
            "model": "mistral:latest",
            "modified_at": "2025-03-05T20:00:00Z",
            "size": 0,
            "digest": "sha256:placeholder"
        },
        {
            "name": "custom:latest",
            "model": "custom:latest",
            "modified_at": "2025-03-05T20:00:00Z",
            "size": 0,
            "digest": "sha256:placeholder"
        }
    ]
    return jsonify({"models": tags}), 200

@app.route('/process_query', methods=['POST'])
async def process_query():
    try:
        data = request.json
        if not data or "query" not in data:
            return jsonify({"error": "Query is required"}), 400

        query = data["query"]
        limit = int(data.get("limit", 5))
        offset = int(data.get("offset", 0))

        cache_key = f"query_cache:{query}:{limit}:{offset}"
        if cached := redis_client.get(cache_key):
            return jsonify({"source": "cache", "response": json.loads(cached)})

        docs = vectorstore.similarity_search(query, k=limit, offset=offset)

        pool = await get_db_pool()
        async with pool.acquire() as conn:
            sql_results = await conn.fetch("""
                SELECT "Номер МД", "Опис товару", "Відправник", "Одержувач", "Код товару"
                FROM customs_data
                WHERE "Опис товару" ILIKE $1 OR "Відправник" ILIKE $1 OR "Одержувач" ILIKE $1 OR "Код товару" ILIKE $1
                LIMIT $2 OFFSET $3
            """, f"%{query}%", limit, offset)

        if not sql_results and not docs:
            return jsonify({"warning": "No results found in SQL or OpenSearch"}), 200

        context = "\n".join(doc.page_content for doc in docs) + "\nSQL Results:\n" + "\n".join(str(row) for row in sql_results)
        truncated_context = context[:MAX_CONTEXT_LENGTH] if len(context) > MAX_CONTEXT_LENGTH else context
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        response = qa_chain({"query": query, "context": truncated_context})

        final_response = {"answer": response["result"], "sources": [doc.metadata for doc in docs] + [{"sql": dict(row)} for row in sql_results]}
        redis_client.setex(cache_key, 3600, json.dumps(final_response))

        await log_query(query, final_response["answer"])
        return jsonify({"source": "rag", "response": final_response})
    except Exception as e:
        logger.error(f"Error in process_query: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
async def chat():
    try:
        data = request.json
        if not data or not isinstance(data.get("messages"), list) or not data["messages"]:
            return jsonify({"error": "Messages must be a non-empty list"}), 400

        query = data["messages"][-1].get("content", "")
        model = data.get("model", "mistral").split(":")[0]  # Видаляємо :latest або інші теги
        limit = int(data.get("limit", 5))
        offset = int(data.get("offset", 0))
        if not query:
            return jsonify({"error": "Query content is required"}), 400

        cache_key = f"chat_cache:{query}:{limit}:{offset}"
        if cached := redis_client.get(cache_key):
            return jsonify(json.loads(cached))

        docs = vectorstore.similarity_search(query, k=limit, offset=offset)
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            sql_results = await conn.fetch("""
                SELECT "Номер МД", "Опис товару", "Відправник", "Одержувач", "Код товару"
                FROM customs_data
                WHERE "Опис товару" ILIKE $1 OR "Відправник" ILIKE $1 OR "Одержувач" ILIKE $1 OR "Код товару" ILIKE $1
                LIMIT $2 OFFSET $3
            """, f"%{query}%", limit, offset)

        if not sql_results and not docs:
            return jsonify({"warning": "No results found in SQL or OpenSearch"}), 200

        context = "\n".join(doc.page_content for doc in docs) + "\nSQL Results:\n" + "\n".join(str(row) for row in sql_results)
        truncated_context = context[:MAX_CONTEXT_LENGTH] if len(context) > MAX_CONTEXT_LENGTH else context
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        response = qa_chain({"query": query, "context": truncated_context})

        chat_response = {
            "model": model,
            "messages": data["messages"],
            "response": response["result"],
            "stream": False,
            "done": True
        }

        redis_client.setex(cache_key, 3600, json.dumps(chat_response))
        await log_query(query, response["result"])
        return jsonify(chat_response)
    except Exception as e:
        logger.error(f"Error in chat: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload_document', methods=['POST'])
async def upload_document():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "File not provided"}), 400

        file = request.files['file']
        if file.content_length > MAX_FILE_SIZE:
            return jsonify({"error": f"File size exceeds {MAX_FILE_SIZE / (1024 * 1024)} MB"}), 400

        filename = file.filename.lower()
        temp_path = f"/tmp/{filename}"
        file.save(temp_path)

        if filename.endswith(".pdf"):
            loader = UnstructuredPDFLoader(temp_path)
        elif filename.endswith((".jpeg", ".jpg", ".png")):
            loader = UnstructuredImageLoader(temp_path)
        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(temp_path)
        elif filename.endswith((".xls", ".xlsx")):
            loader = UnstructuredExcelLoader(temp_path)
        elif filename.endswith(".csv"):
            loader = CSVLoader(temp_path)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        try:
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = text_splitter.split_documents(documents)
            vectorstore.add_texts([doc.page_content for doc in split_docs], metadatas=[{"filename": filename} for _ in split_docs])
        finally:
            os.remove(temp_path)

        return jsonify({"status": "ok", "message": f"File {filename} successfully processed"})
    except Exception as e:
        logger.error(f"Error in upload_document: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/convert_excel_to_csv_from_data', methods=['POST'])
def convert_excel_to_csv_from_data():
    try:
        data = request.json
        logger.info(f"Received request: {data}")
        if not data or "file_name" not in data:
            logger.error("File name is required")
            return jsonify({"error": "File name is required"}), 400

        file_name = data["file_name"]
        input_path = f"/data/{file_name}"
        logger.info(f"Checking file: {input_path}")
        if not os.path.exists(input_path):
            logger.error(f"File not found: {input_path}")
            return jsonify({"error": f"File {input_path} not found"}), 404
        if not input_path.endswith((".xls", ".xlsx")):
            logger.error(f"Unsupported file format: {input_path}")
            return jsonify({"error": "Only .xls or .xlsx files are supported"}), 400

        output_path = f"/data/{os.path.splitext(file_name)[0]}.csv"
        logger.info(f"Converting {input_path} to {output_path}")

        # Очікувані заголовки
        expected_headers = [
            "Час оформлення", "Назва ПМО", "Тип", "Номер МД", "Дата", "Відправник", "ЕДРПОУ", "Одержувач", "№",
            "Код товару", "Опис товару", "Кр.торг.", "Кр.відпр.", "Кр.пох.", "Умови пост.", "Місце пост", "К-ть",
            "Один.вим.", "Брутто, кг.", "Нетто, кг.", "Вага по МД", "ФВ вал.контр", "Особ.перем.", "43", "43_01",
            "РФВ Дол/кг.", "Вага.один.", "Вага різн.", "Контракт", "3001", "3002", "9610", "Торг.марк.",
            "РМВ Нетто Дол/кг.", "РМВ Дол/дод.од.", "РМВ Брутто Дол/кг", "Призн.Зед", "Мін.База Дол/кг.",
            "Різн.мін.база", "КЗ Нетто Дол/кг.", "КЗ Дол/шт.", "Різн.КЗ Дол/кг", "Різ.КЗ Дол/шт",
            "КЗ Брутто Дол/кг.", "Різ.КЗ Брутто", "пільгова", "повна"
        ]

        wb = load_workbook(input_path, read_only=True)
        sheet = wb[wb.sheetnames[0]]

        with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row_count = 0

            # Обробка заголовків
            first_row = next(sheet.rows)
            headers = [cell.value for cell in first_row if cell.value is not None]
            headers = list(dict.fromkeys(headers))  # Видалення дублікатів зі збереженням порядку

            if not headers or len(headers) != len(expected_headers):
                logger.warning(f"Invalid or duplicate headers detected, using expected headers")
                writer.writerow(expected_headers)
            else:
                writer.writerow(headers)

            # Обробка даних
            seen_rows = set()
            first_data_row = True

            for row in sheet.rows:
                row_data = [cell.value for cell in row if cell.value is not None]
                
                # Пропуск порожніх рядків
                if not row_data or all(v is None for v in row_data):
                    continue

                # Перевірка на дублікати
                row_tuple = tuple(row_data)
                if row_tuple not in seen_rows:
                    if first_data_row and headers == expected_headers:
                        first_data_row = False
                        continue

                    writer.writerow(row_data)
                    seen_rows.add(row_tuple)
                    row_count += 1
                    
                    if row_count % 10000 == 0:
                        logger.info(f"Processed {row_count} unique rows")
                else:
                    logger.debug(f"Skipped duplicate row")

        wb.close()
        logger.info(f"Total unique rows converted: {row_count}")

        return send_file(
            output_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"{os.path.splitext(file_name)[0]}.csv"
        )
    except Exception as e:
        logger.error(f"Error in convert_excel_to_csv_from_data: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

async def log_query(query, answer):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("INSERT INTO query_log (query, response) VALUES ($1, $2)", query, answer)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    logger.info("Running Flask app...")
    app.run(host="0.0.0.0", port=5001, debug=True)