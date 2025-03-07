# Інструкція по повному розгортанню та видаленню системи Predator Analytics 2.0

## 1. Загальні вимоги до серверу

### Мінімальні системні вимоги:
- CPU: 8 ядер
- RAM: 32 GB
- Диск: 500 GB SSD
- ОС: Ubuntu 22.04 LTS або новіша / macOS Ventura або новіша
- Docker + Docker Compose
- Відкрити порти:
    - 8080 (API Server)
    - 9200 (OpenSearch)
    - 5432 (PostgreSQL)
    - 6379 (Redis)
    - 3030 (Grafana)
    - 3100 (Loki)

---

## 2. Порядок розгортання

### Крок 1. Завантаження репозиторію
```bash
git clone https://git.yourrepo.com/predator_analytics.git
cd predator_analytics
```

### Крок 2. Налаштування змінних середовища
Створіть `.env` у корені проєкту (приклад додається):
```
POSTGRES_USER=predator
POSTGRES_PASSWORD=securepassword
POSTGRES_DB=customs
OPENSEARCH_HOST=opensearch
REDIS_HOST=redis
LC_API_HOST=langchain_service
```

### Крок 3. Запуск всієї системи
```bash
docker-compose up -d
```

### Крок 4. Перевірка статусу контейнерів
```bash
docker ps
```

### Крок 5. Ініціалізація бази даних PostgreSQL
```bash
docker exec -i predator_postgres psql -U predator -d customs < init/init.sql
```

### Крок 6. Завантаження початкових даних у PostgreSQL та OpenSearch
```bash
python3 ingestion/import_to_opensearch_pg.py --input data/customs_data.json --threads 4
```

### Крок 7. Налаштування Grafana
- Відкрити у браузері: http://localhost:3030
- Логін за замовчуванням: admin / admin
- Імпортувати дашборд з `monitoring/dashboards/predator_dashboard.json`

---

## 3. Корисні команди

### Перезапуск всієї системи
```bash
docker-compose down && docker-compose up -d
```

### Перевірка логів окремого сервісу
```bash
docker logs predator_apiserver -f
```

### Оновлення системи (git pull + оновлення образів)
```bash
git pull

docker-compose down

docker-compose pull

docker-compose up -d
```

---

## 4. Видалення системи та компонентів

### Опція 1 — Повне видалення системи (включно з даними та образами)
```bash
docker-compose down -v --rmi all
```
- Знімає всі контейнери
- Видаляє всі томи (дані PostgreSQL, OpenSearch, Redis)
- Видаляє всі образи (необхідно буде повторно завантажувати)

### Опція 2 — Видалення тільки контейнерів
```bash
docker-compose down
```
- Знімає лише контейнери
- Дані та образи залишаються

### Опція 3 — Видалення кешу та логів
```bash
sudo rm -rf monitoring/logs/*
```

### Опція 4 — Видалення лише образів
```bash
docker-compose down

docker rmi $(docker images 'predator_*' -q)
```

### Опція 5 — Видалення лише даних PostgreSQL, OpenSearch та Redis
```bash
sudo rm -rf data/postgres/*
sudo rm -rf data/opensearch/*
sudo rm -rf data/redis/*
```

### Опція 6 — Очистка всіх невикористаних даних (якщо потрібно глобальне прибирання)
```bash
docker system prune -a --volumes
```

---

## Примітка
Перед виконанням будь-якого виду видалення рекомендовано створити бекап, особливо для PostgreSQL та OpenSearch даних.

Бекап PostgreSQL:
```bash
docker exec predator_postgres pg_dump -U predator -d customs > backup/customs_backup.sql
```

Бекап OpenSearch (приклад через snapshot API):
```bash
curl -XPUT "http://localhost:9200/_snapshot/backup_repository" -H 'Content-Type: application/json' -d'{
  "type": "fs",
  "settings": {
    "location": "/usr/share/opensearch/backup"
  }
}'
```

Далі створити снапшот:
```bash
curl -XPUT "http://localhost:9200/_snapshot/backup_repository/snapshot_1"
```

---

## Загальна рекомендація

Перед повним видаленням завжди проводити перевірку:
- Чи є актуальні резервні копії.
- Чи узгоджено видалення з усіма учасниками проекту.
- Чи не виконується критичне завантаження або індексація.

Цей документ є частиною офіційної технічної документації Predator Analytics 2.0.

.
├── apiserver/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── ingestion/
│   └── import_to_opensearch_pg.py
├── init/
│   └── init.sql
├── data/
│   └── customs_data.json
# Docker Compose Configuration File

## File Structure
This file is located in the root directory of the Predator Analytics 2.0 project.
docker-compose down --rmi all -v --remove-orphans
## Quick Reference
To completely clean up all containers, images, volumes and orphaned containers:
Ось коротка інструкція для розгортання системи Predator Analytics 2.0 (v1.0) та заливки даних із файлу `data/customs_data.json` у бази PostgreSQL та OpenSearch.

---

### Інструкція з розгортання та заливки даних

#### 1. Передумови
- Встановлено **Docker** та **Docker Compose**.
- Наявний файл `data/customs_data.json` у директорії `data/` у корені проекту.
- Усі файли з версії v1.0 (`docker-compose.yml`, `init.sql`, `apiserver/app.py`, `ingestion/import_to_opensearch_pg.py`, `apiserver/Dockerfile`, `apiserver/requirements.txt`) розміщені в правильних директоріях.

#### 2. Підготовка оточення
1. Переконайтеся, що структура директорій виглядає так:
   ```
   .
   ├── apiserver/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── index_pg_os.py
│   
├── init/
│   └── init.sql
├── data/
│   └── vexel.csv
└── docker-compose.yml
   ```
2. Створіть файл `.env` у корені проекту (якщо його немає) із такими змінними:
   ```plaintext
   POSTGRES_USER=predator
   POSTGRES_PASSWORD=strong_password
   POSTGRES_DB=predator_db
   OPENSEARCH_HOSTS=http://opensearch:9200
   REDIS_HOST=redis
   OLLAMA_HOST=http://host.docker.internal:11434
   ```

#### 3. Розгортання системи
1. Відкрийте термінал у кореневій директорії проекту.
2. Виконайте команду для запуску всіх сервісів:
   ```bash
   docker-compose up -d --build
   ```
   - Флаг `-d` запускає контейнери у фоновому режимі.
   - Флаг `--build` забезпечує збірку образів (наприклад, для `apiserver`).
3. Перевірте, чи всі контейнери запущені:
   ```bash
   docker-compose ps
   ```
   Очікуваний результат: усі сервіси (`apiserver`, `postgres`, `redis`, `opensearch`, `openwebui`) у статусі `Up`.

#### 4. Ініціалізація бази PostgreSQL
1. Скопіюйте скрипт `init.sql` у контейнер `postgres`:
   ```bash
   docker cp init/init.sql postgres:/init.sql
   ```
2. Виконайте скрипт для створення таблиць:
   ```bash
   docker exec -i postgres psql -U predator -d predator_db -f /init.sql
   ```

#### 5. Заливка даних із `data/customs_data.json`
1. Переконайтеся, що файл `data/customs_data.json` доступний у директорії `data/`.
2. Запустіть скрипт імпорту у контейнері `apiserver`:
   ```bash
   docker exec -it apiserver python /app/ingestion/import_to_opensearch_pg.py
   ```
   - Скрипт автоматично завантажить дані з `/app/data/customs_data.json` (шлях у контейнері, змонтований через `volumes` у `docker-compose.yml`) у PostgreSQL та OpenSearch.
   - Процес використає 8 потоків і пакети по 5000 записів для швидкості.
3. Дочекайтеся повідомлення в терміналі: `Завантаження завершено`.

#### 6. Перевірка роботи
1. Перевірте доступність Open WebUI:
   - Відкрийте браузер і перейдіть за адресою `http://localhost:3000`.
2. Перевірте API-сервер:
   - Виконайте запит:
     ```bash
     curl http://localhost:5001/health
     ```
     Очікувана відповідь: `{"status": "ok"}`
3. (Опціонально) Перевірте логи для діагностики:
   ```bash
   docker-compose logs apiserver
   ```

#### 7. Зупинка системи (за потреби)
- Для зупинки всіх сервісів:
  ```bash
  docker-compose down
  ```
- Для зупинки з видаленням томів (очищення даних):
  ```bash
  docker-compose down -v
  ```

---

docker-compose down --rmi all -v --remove-orphans
docker system prune -a --volumes
### Примітки
- Якщо `customs_data.json` великий (декілька мільйонів записів), імпорт може зайняти час. Слідкуйте за виводом у терміналі.
- У разі помилок перевірте логи контейнерів (`docker-compose logs <service_name>`).
- Для коректної роботи Ollama переконайтеся, що сервер Ollama доступний за адресою, вказаною в `OLLAMA_HOST`.
docker-compose down
docker-compose build --no-cache
docker-compose up -d
Готово! Система розгорнута, а дані залиті в бази. Ви можете почати використовувати Open WebUI для запитів до API-сервера.