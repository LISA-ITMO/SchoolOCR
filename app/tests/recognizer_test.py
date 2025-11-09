import os
import json
import unittest
from PIL import Image
from app.services.recognizer import DocumentRecognizer


class TestRecognizer(unittest.TestCase):
    def setUp(self):
        self.recognizer = DocumentRecognizer()
        current_dir = os.path.dirname(__file__)
        self.test_data_path = os.path.join(current_dir, "test_data")

    def test_all_subjects(self):
        """Тестирует распознавание для всех предметов в test_data"""
        if not os.path.exists(self.test_data_path):
            self.skipTest(f"Директория {self.test_data_path} не найдена")

        for subject_dir in os.listdir(self.test_data_path):
            subject_path = os.path.join(self.test_data_path, subject_dir)
            if not os.path.isdir(subject_path):
                continue

            with self.subTest(subject=subject_dir):
                self._test_subject_folder(subject_path, subject_dir)

    def _test_subject_folder(self, subject_path, subject_name):
        """Тестирует распознавание для папки предмета"""
        # Находим файлы
        image_file = None
        json_file = None

        for file in os.listdir(subject_path):
            file_path = os.path.join(subject_path, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_file = file_path
            elif file.lower().endswith('.json'):
                json_file = file_path

        if not image_file:
            self.skipTest(f"Не найдено изображение в {subject_name}")
        if not json_file:
            self.skipTest(f"Не найден JSON в {subject_name}")

        # Загружаем ожидаемый результат
        with open(json_file, 'r', encoding='utf-8') as f:
            expected = json.load(f)

        # Распознаем изображение
        image = Image.open(image_file)
        actual = self.recognizer.recognize(image)

        # Сравниваем результаты
        self._compare_results(expected, actual)

    def _compare_results(self, expected, actual):
        """Сравнивает ожидаемый и фактический результаты"""
        # Проверяем основные поля
        # self.assertEqual(expected.get('subject'), actual.get('subject'))
        # self.assertEqual(expected.get('grade'), actual.get('grade'))
        # self.assertEqual(expected.get('variant'), actual.get('variant'))
        # self.assertEqual(expected.get('participant_code'), actual.get('participant_code'))
        self.assertEqual(expected.get('total_score'), actual.get('total_score'))

        # Проверяем оценки по заданиям
        expected_scores = expected.get('scores', {})
        actual_scores = actual.get('scores', {})

        self.assertEqual(len(expected_scores), len(actual_scores))

        for task_num in expected_scores:
            self.assertIn(task_num, actual_scores)

            exp_score, exp_prob = expected_scores[task_num][:2]
            act_score, act_prob, _ = actual_scores[task_num]

            self.assertEqual(exp_score, act_score)
            self.assertAlmostEqual(exp_prob, act_prob, delta=0.2)


if __name__ == '__main__':
    unittest.main()