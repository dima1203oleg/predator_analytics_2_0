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

