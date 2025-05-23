# SchoolOCR
SchoolOCR - api для распознавания структурированных данных в титульных листах всероссийских проверочных работ с применением нейросетевых методов\
Пример титульной страницы:
![sample2](https://github.com/user-attachments/assets/37c95311-d113-4e8f-acbf-c6ce7ed68a10)

# Установка
1. Клонируйте репозиторий:
```bash
git clone https://github.com/LISA-ITMO/SchoolOCR
cd SchoolOCR
```
2. Создайте и активируйте виртуальное окружение:
```bash
# Для Windows
python -m venv venv
venv\Scripts\activate

# Для Linux/macOS
python3 -m venv venv
source venv/bin/activate
```
3. Установите зависимости:
```bash
pip install -r requirements.txt
```
4. Запустите API
```bash
python ./app.py
```
5. Можете сделать тестовый запрос
```bash
python ./scripts/app_interaction/sender.py
```

# Установка с помощью Docker
1. Клонируйте репозиторий:
```bash
git clone https://github.com/LISA-ITMO/SchoolOCR
cd SchoolOCR
```
2. Поднимите контейнер
```bash
docker-compose up --build
```
3. Можете сделать тестовый запрос
```bash
python ./scripts/app_interaction/sender.py
```
# Требования к скану
- Скан, насколько это возможно, сделать ровным и четким;
- Цифры пишутся раздельно, черной (лучше гелиевой) ручкой, без разъединений - в идеале, как печатные. Для таблиц - ровно в клетке, не выходя за рамки;
- Для каждой цифры в таблице - отдельная клетка, границы должны быть как можно четче.
- Передавать API файл в jpg или pdf формате через base64, разрешение исходное
