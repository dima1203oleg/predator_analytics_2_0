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
   ├── ingestion/
   │   └── import_to_opensearch_pg.py
   ├── init/
   │   └── init.sql
   ├── data/
   │   └── customs_data.json
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
1. Скопіюйте скрипт `init.sql` у контейнер:
   ```bash
   docker cp init/init.sql predator_analytics_2_0-postgres-1:/init.sql
   ```
2. Виконайте скрипт для створення таблиць:
   ```bash
   docker exec -i predator_analytics_2_0-postgres-1 psql -U predator -д predator_db -f /init.sql
   ```

#### 5. Заливка даних із `data/customs_data.json`
1. Переконайтеся, що файл `data/customs_data.json` доступний у директорії `data/`.
2. Запустіть скрипт імпорту у контейнері:
   ```bash
   docker exec -it predator_analytics_2_0-apiserver-1 python /app/ingestion/import_to_opensearch_pg.py
   ```
   - Скрипт автоматично завантажить дані з `/app/data/customs_data.json` у PostgreSQL та OpenSearch.
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
   docker-compose logs predator_analytics_2_0-apiserver-1
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

### Примітки
- Якщо `customs_data.json` великий (декілька мільйонів записів), імпорт може зайняти час. Слідкуйте за виводом у терміналі.
- У разі помилок перевірте логи контейнерів (`docker-compose logs <service_name>`).
- Для коректної роботи Ollama переконайтеся, що сервер Ollama доступний за адресою, вказаною в `OLLAMA_HOST`.
docker-compose down
docker-compose build --no-cache
docker-compose up -d
Готово! Система розгорнута, а дані залиті в бази. Ви можете почати використовувати Open WebUI для запитів до API-сервера.