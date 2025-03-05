#!/bin/bash

# Налаштування змінних
PROJECT_DIR="$(pwd)"
CONTAINER_NAME="predator_analytics_2_0-apiserver-1"
POSTGRES_CONTAINER="predator_analytics_2_0-postgres-1"
DATA_FILE="$PROJECT_DIR/data/customs_data.json"
IMPORT_SCRIPT="$PROJECT_DIR/ingestion/import_to_opensearch_pg.py"
INIT_SQL="$PROJECT_DIR/init/init.sql"

# Кольори для виводу
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Функція для перевірки статусу команди
check_status() {
    if [ $? -ne 0 ]; then
        echo -e "${RED}Помилка: $1${NC}"
        exit 1
    else
        echo -e "${GREEN}Успіх: $1${NC}"
    fi
}

# 1. Перевірка наявності необхідних файлів
echo -e "${YELLOW}Перевірка наявності файлів...${NC}"
for file in "$DATA_FILE" "$IMPORT_SCRIPT" "$INIT_SQL"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}Файл $file не знайдено!${NC}"
        exit 1
    fi
done

# 2. Запуск Docker Compose (якщо ще не запущено)
echo -e "${YELLOW}Перевірка та запуск Docker Compose...${NC}"
docker-compose ps | grep "Up" | grep "$CONTAINER_NAME" > /dev/null
if [ $? -ne 0 ]; then
    echo "Запуск Docker Compose..."
    docker-compose up -d
    check_status "Запуск Docker Compose"
    sleep 10  # Чекаємо, поки сервіси піднімуться
else
    echo "Docker Compose уже запущено"
fi

# 3. Перекодування файлів (якщо потрібно, наприклад, у UTF-8)
echo -e "${YELLOW}Перекодування файлів у UTF-8...${NC}"
iconv -f WINDOWS-1251 -t UTF-8 "$DATA_FILE" -o "$DATA_FILE.utf8" 2>/dev/null
if [ $? -eq 0 ]; then
    mv "$DATA_FILE.utf8" "$DATA_FILE"
    echo "Файл $DATA_FILE перекодовано в UTF-8"
else
    echo "Перекодування не потрібне або файл уже в UTF-8"
fi

# 4. Копіювання файлів у контейнер
echo -e "${YELLOW}Копіювання файлів у контейнер...${NC}"
docker cp "$DATA_FILE" "$CONTAINER_NAME:/app/data/customs_data.json"
check_status "Копіювання customs_data.json"
docker cp "$IMPORT_SCRIPT" "$CONTAINER_NAME:/app/ingestion/import_to_opensearch_pg.py"
check_status "Копіювання import_to_opensearch_pg.py"
docker cp "$INIT_SQL" "$POSTGRES_CONTAINER:/tmp/init.sql"
check_status "Копіювання init.sql"

# 5. Перевірка та встановлення залежностей
echo -e "${YELLOW}Перевірка та встановлення залежностей...${NC}"
docker exec "$CONTAINER_NAME" pip install psycopg2-binary==2.9.3 elasticsearch==8.12.0 psutil==5.9.8
check_status "Встановлення залежностей"

# 6. Ініціалізація бази PostgreSQL
echo -e "${YELLOW}Ініціалізація бази PostgreSQL...${NC}"
docker exec -it "$POSTGRES_CONTAINER" psql -U predator -d predator_db -f /tmp/init.sql
check_status "Виконання init.sql"

# 7. Запуск імпорту
echo -e "${YELLOW}Запуск імпорту в PostgreSQL та OpenSearch...${NC}"
docker exec -it "$CONTAINER_NAME" python /app/ingestion/import_to_opensearch_pg.py /app/data/customs_data.json > import_log.txt 2>&1
check_status "Запуск імпорту"

# 8. Контроль процесу імпорту
echo -e "${YELLOW}Контроль результатів імпорту...${NC}"
if grep -q "Завантаження завершено" import_log.txt; then
    echo -e "${GREEN}Імпорт завершено успішно${NC}"
else
    echo -e "${RED}Помилка під час імпорту. Перегляньте import_log.txt${NC}"
    cat import_log.txt
    exit 1
fi

# 9. Тестування підключення до PostgreSQL
echo -e "${YELLOW}Тестування підключення до PostgreSQL...${NC}"
docker exec -it "$CONTAINER_NAME" bash -c "psql -U predator -h postgres -d predator_db -c 'SELECT COUNT(*) FROM customs_data;'" > pg_test.txt 2>&1
if grep -q "count" pg_test.txt; then
    echo -e "${GREEN}PostgreSQL: Дані присутні${NC}"
    cat pg_test.txt
else
    echo -e "${RED}PostgreSQL: Проблема з даними або підключенням${NC}"
    cat pg_test.txt
    exit 1
fi

# 10. Тестування підключення до OpenSearch
echo -e "${YELLOW}Тестування підключення до OpenSearch...${NC}"
docker exec -it "$CONTAINER_NAME" curl -s http://opensearch:9200/customs_data/_count > os_test.txt
if grep -q '"count":' os_test.txt; then
    echo -e "${GREEN}OpenSearch: Дані присутні${NC}"
    cat os_test.txt
else
    echo -e "${RED}OpenSearch: Проблема з даними або підключенням${NC}"
    cat os_test.txt
    exit 1
fi

# 11. Очищення тимчасових файлів
echo -e "${YELLOW}Очищення тимчасових файлів...${NC}"
rm -f import_log.txt pg_test.txt os_test.txt

echo -e "${GREEN}Усі процеси завершені успішно!${NC}"