import requests
import base64
import json

# Адрес сервера
SERVER_URL = "http://158.160.57.65:8000/recognize"

# Путь к изображению
IMAGE_PATH = "../to_proccess/Сканы титульников/химия 8 кл/Химия 8 (1)-1.pdf"

# API-ключ
API_KEY = ""


# Функция для кодирования изображения в base64
def encode_image_to_base64(image_path):
    """
    Encodes an image file to a Base64 string.

      Args:
        image_file: The image file object to be encoded.

      Returns:
        str: A Base64 encoded string representation of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Функция для отправки запроса на сервер
def send_image_to_server(image_base64):
    """
    Sends image data to a server and returns the response.

        Args:
            payload: The payload containing the image data, expected to be a dictionary
                     that can be serialized into JSON.

        Returns:
            dict: A dictionary representing the JSON response from the server if the request
                  was successful (status code 200).  None is returned if the request fails.
    """
    # Формируем JSON-тело запроса
    payload = {"image_base64": image_base64}

    # Заголовки с API-ключом
    headers = {"Authorization": API_KEY}

    # Отправляем POST-запрос на сервер
    response = requests.post(SERVER_URL, json=payload, headers=headers)

    # Проверяем статус ответа
    if response.status_code == 200:
        return response.json()  # Возвращаем JSON-ответ
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.text)
        return None


# Основная функция
def main():
    """
    Sends an image to a server and prints the response.

        This function reads an image from a specified path, encodes it as a base64 string,
        sends it to a server using the send_image_to_server function, and then prints
        the JSON formatted response from the server if successful.

        Parameters:
            None

        Returns:
            None
    """
    # Кодируем изображение в base64
    image_base64 = encode_image_to_base64(IMAGE_PATH)
    print("Изображение успешно закодировано в base64.")

    # Отправляем изображение на сервер
    print("Отправка изображения на сервер...")
    result = send_image_to_server(image_base64)

    # Выводим результат
    if result:
        print("Ответ от сервера:")
        print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
