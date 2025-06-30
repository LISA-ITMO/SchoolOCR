import os
import json
import cv2
import unittest
from services.recognition_service import RecognitionService

TEST_DATA_DIR = "./test_data"


class TestRecognitionService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Инициализация сервиса перед всеми тестами"""
        cls.service = RecognitionService(
            config_path="../config.json",
            api_keys_path="../api_keys.json",
            models_path="../models"
        )
        cls.test_cases = cls.load_test_cases()

    @staticmethod
    def load_test_cases():
        """Загрузка тестовых случаев"""
        test_cases = []
        for subject in os.listdir(TEST_DATA_DIR):
            subject_dir = os.path.join(TEST_DATA_DIR, subject)
            if os.path.isdir(subject_dir):
                for file in os.listdir(subject_dir):
                    if file.endswith(".json"):
                        img_file = file.replace(".json", ".jpg")
                        img_path = os.path.join(subject_dir, img_file)
                        json_path = os.path.join(subject_dir, file)

                        if os.path.exists(img_path):
                            with open(json_path, "r", encoding="utf-8") as f:
                                expected = json.load(f)
                            test_cases.append((img_path, expected))
        return test_cases

    def test_all_images(self):
        """Тестирование всех изображений"""
        print(len(self.test_cases))
        for img_path, expected in self.test_cases:
            with self.subTest(img_path=img_path):
                # Чтение и подготовка изображения
                img = cv2.imread(img_path)
                self.assertIsNotNone(img, f"Не удалось загрузить изображение: {img_path}")

                _, img_encoded = cv2.imencode('.jpg', img)
                img_bytes = img_encoded.tobytes()

                # Обработка изображения
                result = self.service.recognize_image(img_bytes)

                # Проверки
                self.assertIsNotNone(result, "Сервис вернул None")
                self.assertIn("subject", result, "Отсутствует поле subject")
                self.assertIn("grade", result, "Отсутствует поле grade")

                if expected.get("participant_code"):
                    self.assertEqual(
                        result["participant_code"],
                        expected["participant_code"],
                        f"Неверный код участника для {img_path}"
                    )


if __name__ == '__main__':
    unittest.main()