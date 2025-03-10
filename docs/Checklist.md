# Predator Analytics 2.0 — Фінальний чекліст перед запуском

## 1. Перевірка компонентів

### Перевірка Open WebUI
- [x] Перевірено підключення до API Server
- [x] Відображаються всі основні розділи
- [x] Логін, авторизація, налаштування запитів працюють коректно

### Перевірка API Server
- [x] API Server стартує без помилок
- [x] Коректно приймає запити від Open WebUI
- [x] Коректно звертається до LangChain Service
- [x] Коректно звертається до OpenSearch
- [x] Коректно звертається до PostgreSQL
- [x] Логування відбувається в Loki
- [x] Обробка помилок налаштована

### Перевірка LangChain Service
- [x] Контейнер LangChain Service запускається без помилок
- [x] LangChain отримує запити від API Server
- [x] Правильно завантажуються моделі через Ollama (або інший обраний бекенд)
- [x] Всі залежності LangChain встановлені (requirements.txt)
- [x] Всі критичні події логуються (помилки, таймаути, некоректні відповіді)
- [x] LangChain може правильно формувати векторні запити до OpenSearch
- [x] Логи LangChain Service відображаються в Grafana через Loki

### Перевірка PostgreSQL
- [x] Контейнер PostgreSQL стартує без помилок
- [x] Всі таблиці створені відповідно до init.sql
- [x] Встановлено пул з’єднань (наприклад, через psycopg2 connection pooling)
- [x] Логування запитів та помилок активне
- [x] Перевірено з'єднання з API Server

### Перевірка OpenSearch
- [x] Контейнер OpenSearch стартує без помилок
- [x] Індекси створені та відповідають JSON-структурі декларацій
- [x] Коректна взаємодія з LangChain Service
- [x] Коректна взаємодія з API Server
- [x] Логуються всі критичні події (падіння, помилки індексації)

### Перевірка Redis
- [x] Контейнер Redis стартує без помилок
- [x] Кешування працює
- [x] Встановлено TTL
- [x] Коректна взаємодія з API Server
- [x] Логуються всі збої підключення

### Перевірка Data Ingestion Service
- [x] Контейнер Ingestion Service стартує без помилок
- [x] Дані заливаються одночасно в OpenSearch та PostgreSQL
- [x] Підтримка паралельних потоків працює
- [x] Логи потоку зберігаються у Loki
- [x] Помилки імпорту логуються та відображаються в Grafana

### Перевірка Grafana + Loki + Promtail
- [x] Контейнери стартують без помилок
- [x] Promtail читає логи зі всіх контейнерів
- [x] Loki отримує всі логи
- [x] Grafana правильно відображає всі панелі
- [x] Алерти налаштовані (при необхідності)

---

## 2. Перевірка мережевих з’єднань
- [x] Всі сервіси у єдиній внутрішній мережі Docker Compose
- [x] API Server бачить LangChain Service, OpenSearch, PostgreSQL та Redis
- [x] LangChain Service бачить OpenSearch
- [x] Promtail має доступ до Docker API (для збору логів)
- [x] OpenSearch доступний за внутрішньою адресою в мережі Docker
- [x] PostgreSQL доступний за внутрішньою адресою в мережі Docker
- [x] Redis доступний за внутрішньою адресою в мережі Docker
- [x] Grafana бачить Loki

---

## 3. Перевірка налаштувань

### OpenSearch
- [x] Налаштовані правильні індекси та мапінги під JSON-декларації
- [x] Встановлений репліка-фактор та shard configuration під реальне навантаження
- [x] Права доступу та безпека налаштовані (якщо security plugin активний)
- [x] Налаштовані ретраї при збої підключення
- [x] Логи критичних помилок йдуть в Loki

### PostgreSQL
- [x] Всі таблиці створені відповідно до init.sql
- [x] Встановлено пул з’єднань (наприклад, через psycopg2 connection pooling)
- [x] Логування запитів та помилок активне

### Redis
- [x] Максимальний розмір кешу встановлений
- [x] Встановлена TTL для кешованих відповідей

### API Server
- [x] Обробка помилок налаштована (збої PostgreSQL, OpenSearch, LangChain)
- [x] Логування рівня INFO/ERROR налаштоване
- [x] Встановлена підтримка health-check ендпоїнта

---

## 4. Перевірка продуктивності та потоків
- [x] Data Ingestion Service налаштований на кількість потоків, адаптовану до серверних ресурсів
- [x] Перевірено налаштування потоків за замовчуванням для Mac Studio M1 Max
- [x] Передбачено можливість кастомного налаштування потоків через конфіг

---

## 5. Перевірка Grafana Dashboard
- [x] Підключено джерело даних Loki
- [x] Імпортовано оновлений дашборд `predator_dashboard.json`
- [x] Всі панелі коректно відображають інформацію:
    - Загальні запити по контейнерах
    - Помилки по контейнерах
    - Частота запитів
    - Кеш-хіти Redis
    - Індексація OpenSearch
    - Помилки LangChain Service
    - Збої підключень до PostgreSQL, OpenSearch, Redis

---

## 6. Перевірка стратегії відмовостійкості
- [x] Встановлено ретраї при відмовах підключення до баз даних
- [x] Логуються всі збої при обробці запитів
- [x] Збої критичних сервісів (OpenSearch, PostgreSQL) відображаються у Grafana

---

## 7. Контрольна перевірка developer_guide.md
- [x] Остання версія файлу `developer_guide.md` переглянута
- [x] Всі зміни по інфраструктурі, структурі даних, потоках заливок, моніторингу внесені
- [x] Файл синхронізований із поточною версією системи

---

## 8. Загальний статус
✅ **Система готова до запуску у продуктивне середовище (після тестування)**

---

## Примітка
Цей чекліст необхідно виконувати при кожному серйозному оновленні інфраструктури або функціоналу системи.

