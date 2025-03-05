#!/bin/bash

PROJECT_DIR="$(pwd)"
CONTAINER_NAME="predator_analytics_2_0-apiserver-1"
POSTGRES_CONTAINER="predator_analytics_2_0-postgres-1"
TEMP_DATA_FILE="$PROJECT_DIR/data/customs_data.json"
DATA_FILE="$PROJECT_DIR/data/customs_data_fixed.json"
IMPORT_SCRIPT="$PROJECT_DIR/ingestion/import_to_opensearch_pg.py"
INIT_SQL="$PROJECT_DIR/init/init.sql"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_status() {
    if [ $? -ne 0 ]; then
        echo -e "${RED}Помилка: $1${NC}"
        if [ -f import_log.txt ]; then
            echo -e "${RED}Деталі помилки:${NC}"
            cat import_log.txt
        fi
        exit 1
    else
        echo -e "${GREEN}Успіх: $1${NC}"
    fi
}

echo -e "${YELLOW}Перевірка наявності файлів...${NC}"
for file in "$TEMP_DATA_FILE" "$IMPORT_SCRIPT" "$INIT_SQL"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}Файл $file не знайдено!${NC}"
        exit 1
    fi
done

echo -e "${YELLOW}Виправлення JSON...${NC}"
python fix_json.py "$TEMP_DATA_FILE" "$DATA_FILE"
check_status "Виправлення customs_data.json"

echo -e "${YELLOW}Перевірка та запуск Docker Compose...${NC}"
docker-compose ps | grep "Up" | grep "$CONTAINER_NAME" > /dev/null
if [ $? -ne 0 ]; then
    echo "Запуск Docker Compose..."
    docker-compose up -d
    check_status "Запуск Docker Compose"
    sleep 10
else
    echo "Docker Compose уже запущено"
fi

echo -e "${YELLOW}Копіювання файлів у контейнер...${NC}"
docker cp "$DATA_FILE" "$CONTAINER_NAME:/app/data/customs_data_fixed.json"
check_status "Копіювання customs_data_fixed.json"
docker cp "$IMPORT_SCRIPT" "$CONTAINER_NAME:/app/ingestion/import_to_opensearch_pg.py"
check_status "Копіювання import_to_opensearch_pg.py"
docker cp "$INIT_SQL" "$POSTGRES_CONTAINER:/tmp/init.sql"
check_status "Копіювання init.sql"

echo -e "${YELLOW}Перевірка та встановлення залежностей...${NC}"
docker exec "$CONTAINER_NAME" pip install psycopg2-binary==2.9.3 opensearch-py==2.3.0 psutil==5.9.8 ijson==3.3.0
check_status "Встановлення залежностей"

echo -e "${YELLOW}Ініціалізація бази PostgreSQL...${NC}"
docker exec -it "$POSTGRES_CONTAINER" psql -U predator -d predator_db -f /tmp/init.sql
check_status "Виконання init.sql"

echo -e "${YELLOW}Запуск імпорту в PostgreSQL та OpenSearch...${NC}"
docker exec -it "$CONTAINER_NAME" python /app/ingestion/import_to_opensearch_pg.py /app/data/customs_data_fixed.json > import_log.txt 2>&1
check_status "Запуск імпорту"

echo -e "${YELLOW}Контроль результатів імпорту...${NC}"
if grep -q "Завантаження завершено" import_log.txt; then
    echo -e "${GREEN}Імпорт завершено успішно${NC}"
else
    echo -e "${RED}Помилка під час імпорту. Перегляньте import_log.txt${NC}"
    cat import_log.txt
    exit 1
fi

echo -e "${YELLOW}Тестування підключення до PostgreSQL...${NC}"
docker exec -it "$CONTAINER_NAME" python -c "import psycopg2; conn = psycopg2.connect(dbname='predator_db', user='predator', password='strong_password', host='postgres'); cur = conn.cursor(); cur.execute('SELECT COUNT(*) FROM customs_data'); print(cur.fetchone()[0]); conn.close()" > pg_test.txt 2>&1
if grep -q "[0-9]" pg_test.txt; then
    echo -e "${GREEN}PostgreSQL: Дані присутні${NC}"
    cat pg_test.txt
else
    echo -e "${RED}PostgreSQL: Проблема з даними або підключенням${NC}"
    cat pg_test.txt
    exit 1
fi

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

echo -e "${YELLOW}Очищення тимчасових файлів...${NC}"
rm -f import_log.txt pg_test.txt os_test.txt

echo -e "${GREEN}Усі процеси завершені успішно!${NC}"