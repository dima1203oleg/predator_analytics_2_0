import sys
import re
import ijson
import json
import decimal

class DecimalEncoder(json.JSONEncoder):
    """Кастомний енкодер для обробки Decimal."""
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        return super().default(obj)

def clean_key(key):
    """Очищає ключ від зайвих пробілів і символів."""
    cleaned = key.strip()
    if cleaned.endswith('м') and ' ' in cleaned:
        cleaned = cleaned[:cleaned.rindex(' ')]
    return cleaned.strip()

def preprocess_json(input_file, temp_file):
    """Попередньо виправляє невалідний JSON, замінюючи NaN на null."""
    print("Попередня обробка JSON: заміна NaN на null...")
    with open(input_file, 'r', encoding='utf-8') as infile, open(temp_file, 'w', encoding='utf-8') as outfile:
        content = infile.read()
        # Замінюємо NaN на null, враховуючи можливі контексти
        fixed_content = re.sub(r'(?<=:|,)\s*NaN\s*(?=,|\n|})', ' null', content)
        outfile.write(fixed_content)
    print(f"Попередньо виправлений JSON збережено в {temp_file}")

def fix_json(input_file, output_file):
    temp_file = input_file + '.tmp'
    
    # Крок 1: Виправляємо NaN у файлі
    preprocess_json(input_file, temp_file)
    
    # Крок 2: Обробляємо JSON із виправленням ключів і Decimal
    with open(temp_file, 'rb') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        print("Починаємо обробку JSON...")
        outfile.write('[')  # Починаємо масив
        parser = ijson.items(infile, 'item')
        first = True
        count = 0
        for item in parser:
            # Очищаємо ключі в словнику
            cleaned_item = {clean_key(k): v for k, v in item.items()}
            if not first:
                outfile.write(',')
            else:
                first = False
            json.dump(cleaned_item, outfile, ensure_ascii=False, cls=DecimalEncoder)
            count += 1
            if count % 1000 == 0:  # Логування прогресу
                print(f"Оброблено {count} записів")
        outfile.write(']')  # Завершуємо масив
    
    # Видаляємо тимчасовий файл
    import os
    os.remove(temp_file)
    
    print(f"Виправлений JSON збережено в {output_file}. Усього оброблено {count} записів")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Використання: python fix_json.py <input_file> <output_file>")
        sys.exit(1)
    fix_json(sys.argv[1], sys.argv[2])