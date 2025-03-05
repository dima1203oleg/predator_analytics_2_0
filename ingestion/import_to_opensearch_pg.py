import os
import json
import sys
import psycopg2
from opensearchpy import OpenSearch, helpers
from psutil import cpu_count
import multiprocessing
import ijson
from tenacity import retry, stop_after_attempt, wait_exponential
import decimal
from langchain_community.embeddings import OllamaEmbeddings

PG_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "predator_db"),
    "user": os.getenv("POSTGRES_USER", "predator"),
    "password": os.getenv("POSTGRES_PASSWORD", "strong_password"),
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": os.getenv("POSTGRES_PORT", 5432)
}

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOSTS", "http://opensearch:9200")
INDEX_NAME = "customs_data"
os_client = OpenSearch([OPENSEARCH_HOST], http_auth=("admin", "strong_password"), use_ssl=False)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
embeddings = OllamaEmbeddings(model="mistral", base_url=OLLAMA_HOST)

def clean_key(key):
    cleaned = key.strip()
    if cleaned.endswith('м') and ' ' in cleaned:
        cleaned = cleaned[:cleaned.rindex(' ')]
    return cleaned.strip()

def clean_value(value, expected_type):
    if value is None or value == "":
        return None
    try:
        if expected_type == "TIMESTAMP":
            return str(value) if value else None
        elif expected_type in ("BIGINT", "INT"):
            if isinstance(value, (int, float)):
                return int(value)
            elif isinstance(value, str) and value.strip().isdigit():
                return int(value)
            else:
                print(f"Пропущено значення {value} для типу {expected_type} (не є числом)")
                return None
        elif expected_type == "DOUBLE PRECISION":
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                return float(value)
            else:
                print(f"Пропущено значення {value} для типу {expected_type} (не є числом)")
                return None
        elif expected_type == "TEXT":
            return str(value) if value else None
    except (ValueError, TypeError) as e:
        print(f"Помилка перетворення значення {value} у тип {expected_type}: {e}")
        return None
    return value

def clean_data(data):
    if isinstance(data, dict):
        return {clean_key(k): clean_data(v) for k, v in data.items()}
    elif isinstance(data, float) and (data != data):
        return None
    elif isinstance(data, decimal.Decimal):
        return float(data)
    elif isinstance(data, list):
        return [clean_data(item) for item in data]
    return data

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def insert_to_postgres(data_batch):
    print(f"Вставка {len(data_batch)} записів у PostgreSQL")
    conn = psycopg2.connect(**PG_CONFIG)
    cursor = conn.cursor()
    query = """
        INSERT INTO customs_data (
            "Час оформлення", "Назва ПМО", "Тип", "Номер МД", "Дата",
            "Відправник", "ЕДРПОУ", "Одержувач", "№", "Код товару",
            "Опис товару", "Кр.торг.", "Кр.відпр.", "Кр.пох.", "Умови пост.",
            "Місце пост", "К-ть", "Один.вим.", "Брутто, кг.", "Нетто, кг.",
            "Вага по МД", "ФВ вал.контр", "Особ.перем.", "43", "43_01",
            "РФВ Дол/кг.", "Вага.один.", "Вага різн.", "Контракт", "3001",
            "3002", "9610", "Торг.марк.", "РМВ Нетто Дол/кг.", "РМВ Дол/дод.од.",
            "РМВ Брутто Дол/кг", "Призн.Зед", "Мін.База Дол/кг.", "Різн.мін.база",
            "КЗ Нетто Дол/кг.", "КЗ Дол/шт.", "Різн.КЗ Дол/кг", "Різ.КЗ Дол/шт",
            "КЗ Брутто Дол/кг.", "Різ.КЗ Брутто", "пільгова", "повна"
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
    """
    column_types = [
        "TIMESTAMP", "TEXT", "TEXT", "TEXT", "TIMESTAMP",
        "TEXT", "BIGINT", "TEXT", "INT", "TEXT",
        "TEXT", "TEXT", "TEXT", "TEXT", "TEXT",
        "TEXT", "DOUBLE PRECISION", "INT", "DOUBLE PRECISION", "DOUBLE PRECISION",
        "DOUBLE PRECISION", "DOUBLE PRECISION", "TEXT", "INT", "INT",
        "DOUBLE PRECISION", "DOUBLE PRECISION", "DOUBLE PRECISION", "TEXT", "INT",
        "INT", "INT", "TEXT", "DOUBLE PRECISION", "DOUBLE PRECISION",
        "DOUBLE PRECISION", "INT", "DOUBLE PRECISION", "DOUBLE PRECISION",
        "DOUBLE PRECISION", "DOUBLE PRECISION", "DOUBLE PRECISION", "DOUBLE PRECISION",
        "DOUBLE PRECISION", "DOUBLE PRECISION", "INT", "INT"
    ]

    records = []
    for d in (clean_data(d) for d in data_batch):
        try:
            record = tuple(clean_value(d.get(col), col_type) for col, col_type in zip(
                ["Час оформлення", "Назва ПМО", "Тип", "Номер МД", "Дата",
                 "Відправник", "ЕДРПОУ", "Одержувач", "№", "Код товару",
                 "Опис товару", "Кр.торг.", "Кр.відпр.", "Кр.пох.", "Умови пост.",
                 "Місце пост", "К-ть", "Один.вим.", "Брутто, кг.", "Нетто, кг.",
                 "Вага по МД", "ФВ вал.контр", "Особ.перем.", "43", "43_01",
                 "РФВ Дол/кг.", "Вага.один.", "Вага різн.", "Контракт", "3001",
                 "3002", "9610", "Торг.марк.", "РМВ Нетто Дол/кг.", "РМВ Дол/дод.од.",
                 "РМВ Брутто Дол/кг", "Призн.Зед", "Мін.База Дол/кг.", "Різн.мін.база",
                 "КЗ Нетто Дол/кг.", "КЗ Дол/шт.", "Різн.КЗ Дол/кг", "Різ.КЗ Дол/шт",
                 "КЗ Брутто Дол/кг.", "Різ.КЗ Брутто", "пільгова", "повна"],
                column_types
            ))
            records.append(record)
        except Exception as e:
            print(f"Помилка обробки запису для PostgreSQL: {d}, помилка: {e}")
            continue

    try:
        cursor.executemany(query, records)
        conn.commit()
    except Exception as e:
        print(f"Помилка вставки в PostgreSQL: {e}")
        raise
    finally:
        cursor.close()
        conn.close()
    print(f"Успішно вставлено {len(records)} записів у PostgreSQL")

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def insert_to_opensearch(data_batch):
    if not os_client.ping():
        raise ConnectionError("Не вдалося підключитися до OpenSearch")
    print(f"Вставка {len(data_batch)} записів у OpenSearch")
    actions = []
    for d in (clean_data(d) for d in data_batch):
        description = d.get("Опис товару", "") or ""
        # Обрізаємо до 2048 символів
        truncated_description = description[:2048] if len(description) > 2048 else description
        vector = embeddings.embed_query(truncated_description) if truncated_description else []
        try:
            actions.append({
                "_index": INDEX_NAME,
                "_source": {
                    **d,
                    "vector": vector
                }
            })
        except Exception as e:
            print(f"Помилка підготовки запису для OpenSearch: {d}, помилка: {e}")
            continue
    
    successful_count = 0
    try:
        response = helpers.bulk(os_client, actions, refresh=True, request_timeout=60, raise_on_error=False)
        successful_count = response[0]
        if response[1]:
            print(f"Виявлено помилки індексації в OpenSearch:")
            for error in response[1][:5]:
                print(f"Деталі помилки: {error}")
    except Exception as e:
        print(f"Помилка вставки в OpenSearch: {e}")
        raise
    
    print(f"Успішно вставлено {successful_count} записів із {len(actions)} у OpenSearch")

def process_batch(batch):
    print(f"Починаємо обробку пакета з {len(batch)} записів")
    insert_to_postgres(batch)
    insert_to_opensearch(batch)
    print(f"Оброблено {len(batch)} записів")

def load_data_from_json(json_path, batch_size=1000, num_workers=None):
    print(f"Спроба відкрити файл: {json_path}")
    num_workers = num_workers or cpu_count(logical=False)
    data_batch = []
    total_records = 0

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            print(f"Файл {json_path} відкрито успішно")
            parser = ijson.items(file, "item")
            for record in parser:
                data_batch.append(record)
                total_records += 1
                if len(data_batch) >= batch_size:
                    process_batch(data_batch)
                    data_batch = []
                    print(f"Загалом оброблено {total_records} записів")
            if data_batch:
                process_batch(data_batch)
                print(f"Загалом оброблено {total_records} записів")
    except Exception as e:
        print(f"Помилка при зчитуванні або обробці JSON: {e}")
        raise

    print(f"Завантаження завершено. Усього оброблено {total_records} записів")

if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else "customs_data.json"
    try:
        load_data_from_json(json_path, batch_size=1000, num_workers=8)
    except Exception as e:
        print(f"Критична помилка в імпорті: {e}")
        sys.exit(1)