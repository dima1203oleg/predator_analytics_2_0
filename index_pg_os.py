import requests
from tqdm import tqdm
import json
import time

FLASK_URL = "http://localhost:5001/index_csv"
FILE_NAME = "vexel.csv"

response = requests.post(
    FLASK_URL,
    json={"file_name": FILE_NAME},
    headers={"Content-Type": "application/json"},
    stream=True,
    timeout=60  # Збільшено тайм-аут до 60 секунд
)
print("Response status:", response.status_code)
print("Response headers:", response.headers)

total_rows = 0
with tqdm(total=total_rows, desc="Indexing Progress", unit="rows", dynamic_ncols=True) as pbar:
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            print("Raw line:", decoded_line)
            try:
                data = json.loads(decoded_line)
                print("Parsed data:", data)
                if "total_rows" in data:
                    total_rows = data["total_rows"]
                    pbar.total = total_rows
                elif "progress" in data:
                    pbar.update(data["progress"] - pbar.n)
                elif "status" in data:
                    print(f"Completed: {data['message']}")
                    # Не завершуємо цикл одразу, чекаємо ще кілька ітерацій
                elif "error" in data:
                    print(f"Error: {data['error']}")
                    break
            except json.JSONDecodeError as e:
                print(f"Failed to parse line: {decoded_line}, error: {e}")
        else:
            print("Empty line received")
            time.sleep(0.5)  # Збільшено затримку до 0.5 секунди для повільнішого потоку

print("Streaming finished")