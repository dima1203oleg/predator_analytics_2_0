import os
import json
import sys
import psycopg2
from opensearchpy import OpenSearch, helpers
from psutil import cpu_count
import multiprocessing
from tenacity import retry, stop_after_attempt, wait_exponential

PG_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "predator_db"),
    "user": os.getenv("POSTGRES_USER", "predator"),
    "password": os.getenv("POSTGRES_PASSWORD", "strong_password"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", 5432)
}

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOSTS", "http://localhost:9200")
INDEX_NAME = "customs_data"
os_client = OpenSearch([OPENSEARCH_HOST], http_auth=("admin", "strong_password"), use_ssl=False)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def insert_to_postgres(data_batch):
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
    records = [(
        d["Час оформлення"], d["Назва ПМО"], d["Тип"], d["Номер МД"], d["Дата"],
        d["Відправник"], d["ЕДРПОУ"], d["Одержувач"], d["№"], d["Код товару"],
        d["Опис товару"], d["Кр.торг."], d["Кр.відпр."], d["Кр.пох."], d["Умови пост."],
        d["Місце пост"], d["К-ть"], d["Один.вим."], d["Брутто, кг."], d["Нетто, кг."],
        d["Вага по МД"], d["ФВ вал.контр"], d["Особ.перем."], d["43"], d["43_01"],
        d["РФВ Дол/кг."], d["Вага.один."], d["Вага різн."], d["Контракт"], d["3001"],
        d["3002"], d["9610"], d["Торг.марк."], d["РМВ Нетто Дол/кг."], d["РМВ Дол/дод.од."],
        d["РМВ Брутто Дол/кг"], d["Призн.Зед"], d["Мін.База Дол/кг."], d["Різн.мін.база"],
        d["КЗ Нетто Дол/кг."], d["КЗ Дол/шт."], d["Різн.КЗ Дол/кг"], d["Різ.КЗ Дол/шт"],
        d["КЗ Брутто Дол/кг."], d["Різ.КЗ Брутто"], d["пільгова"], d["повна"]
    ) for d in data_batch]
    cursor.executemany(query, records)
    conn.commit()
    cursor.close()
    conn.close()

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def insert_to_opensearch(data_batch):
    actions = [{"_index": INDEX_NAME, "_source": d} for d in data_batch]
    helpers.bulk(os_client, actions, refresh=True, request_timeout=60)

def process_batch(batch):
    insert_to_postgres(batch)
    insert_to_opensearch(batch)
    print(f"Оброблено {len(batch)} записів")

def load_data_from_json(json_path, batch_size=1000, num_workers=None):
    print(f"Спроба відкрити файл: {json_path}")
    num_workers = num_workers or cpu_count(logical=False)
    with open(json_path, "r", encoding="utf-8") as file:
        print(f"Файл {json_path} відкрито успішно")
        data = json.load(file)
    pool = multiprocessing.Pool(num_workers)
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        pool.apply_async(process_batch, (batch,))
    pool.close()
    pool.join()
    print("Завантаження завершено")

if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else "customs_data.json"
    load_data_from_json(json_path, batch_size=5000, num_workers=8)