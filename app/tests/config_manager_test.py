import unittest
import json
import os
import tempfile
import shutil
from app.services.config_manager import ConfigManager


class TestConfigManager(unittest.TestCase):

    def setUp(self):
        """Создает временную директорию и файл конфига для тестов"""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_config.json")
        self.config_manager = ConfigManager(self.config_path)

    def tearDown(self):
        """Удаляет временную директорию после тестов"""
        shutil.rmtree(self.test_dir)


    def test_initialization_with_custom_path(self):
        """Тест инициализации с кастомным путем"""
        config = ConfigManager(self.config_path)
        self.assertEqual(config.config_path, self.config_path)

    def test_create_default_config(self):
        """Тест создания дефолтного конфига"""
        config = self.config_manager.get_all_config()

        # Проверяем обязательные секции
        self.assertIn('regions', config)
        self.assertIn('default', config)

        # Проверяем структуру регионов
        regions = config['regions']
        self.assertIn('hat', regions)
        self.assertIn('code', regions)
        self.assertIn('hat_reserve', regions)

        # Проверяем дефолтный score_range
        self.assertEqual(config['default']['score_range'], [0, 5])

    def test_get_regions_config(self):
        """Тест получения конфигурации регионов"""
        regions = self.config_manager.get_regions_config()
        self.assertIsInstance(regions, dict)
        self.assertIn('hat', regions)

        hat_region = regions['hat']
        self.assertEqual(hat_region['x1'], 0)
        self.assertEqual(hat_region['y1'], 0)
        self.assertEqual(hat_region['x2'], 1489)
        self.assertEqual(hat_region['y2'], 400)

    def test_get_region_coordinates(self):
        """Тест получения координат региона"""
        coords = self.config_manager.get_region_coordinates('hat')
        self.assertEqual(coords, (0, 0, 1489, 400))

        # Несуществующий регион
        coords = self.config_manager.get_region_coordinates('nonexistent')
        self.assertIsNone(coords)

    def test_update_regions_config(self):
        """Тест обновления конфигурации регионов"""
        new_regions = {
            'hat': {'x1': 10, 'y1': 20, 'x2': 100, 'y2': 200},
            'code': {'x1': 100, 'y1': 20, 'x2': 200, 'y2': 200}
        }

        self.config_manager.update_regions_config(new_regions)

        regions = self.config_manager.get_regions_config()
        self.assertEqual(regions['hat']['x1'], 10)
        self.assertEqual(regions['hat']['y1'], 20)

    def test_update_region_coordinates(self):
        """Тест обновления координат региона"""
        self.config_manager.update_region_coordinates('hat', 5, 10, 100, 150)

        coords = self.config_manager.get_region_coordinates('hat')
        self.assertEqual(coords, (5, 10, 100, 150))

        # Проверяем что сохранилось в файле
        with open(self.config_path, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)
            self.assertEqual(saved_config['regions']['hat']['x1'], 5)

    def test_set_score_range(self):
        """Тест установки score_range"""
        self.config_manager.set_score_range('физика', '10', (0, 4))

        score_range = self.config_manager.get_score_range('физика', '10')
        self.assertEqual(score_range, (0, 4))

    def test_update_subject_config(self):
        """Тест обновления конфигурации предмета"""
        new_math_config = {
            '7': {'score_range': [0, 5]},
            '8': {'score_range': [0, 6]},
            '10': {'score_range': [0, 10]}
        }

        self.config_manager.update_subject_config('математика', new_math_config)

        config = self.config_manager.get_subject_config('математика')
        self.assertEqual(config['7']['score_range'], [0, 5])
        self.assertEqual(config['10']['score_range'], [0, 10])

    def test_update_grade_config(self):
        """Тест обновления конфигурации класса"""
        new_grade_config = {'score_range': [0, 7]}

        self.config_manager.update_grade_config('математика', '11', new_grade_config)

        grade_config = self.config_manager.get_grade_config('математика', '11')
        self.assertEqual(grade_config['score_range'], [0, 7])

    def test_delete_subject(self):
        """Тест удаления предмета"""
        # Сначала добавляем предмет
        self.config_manager.set_score_range('химия', '8', (0, 3))

        # Убеждаемся что предмет есть
        self.assertIn('химия', self.config_manager.list_subjects())

        # Удаляем предмет
        self.config_manager.delete_subject('химия')

        # Проверяем что предмет удален
        self.assertNotIn('химия', self.config_manager.list_subjects())

    def test_delete_protected_subject(self):
        """Тест попытки удаления защищенных предметов"""
        # Попытка удалить regions
        self.config_manager.delete_subject('regions')
        self.assertIn('regions', self.config_manager.get_all_config())

        # Попытка удалить default
        self.config_manager.delete_subject('default')
        self.assertIn('default', self.config_manager.get_all_config())

    def test_delete_grade_removes_empty_subject(self):
        """Тест что пустой предмет удаляется при удалении последнего класса"""
        # Добавляем предмет с одним классом
        self.config_manager.set_score_range('астрономия', '11', (0, 2))

        # Удаляем единственный класс
        self.config_manager.delete_grade('астрономия', '11')

        # Проверяем что предмет тоже удален
        self.assertNotIn('астрономия', self.config_manager.list_subjects())

    def test_list_subjects(self):
        """Тест получения списка предметов"""
        subjects = self.config_manager.list_subjects()

        # Защищенные ключи не должны быть в списке
        self.assertNotIn('regions', subjects)
        self.assertNotIn('default', subjects)


    def test_update_all_config(self):
        """Тест полного обновления конфига"""
        new_config = {
            'regions': {
                'hat': {'x1': 1, 'y1': 1, 'x2': 100, 'y2': 50}
            },
            'default': {
                'score_range': [0, 10]
            },
            'физика': {
                '10': {'score_range': [0, 5]}
            }
        }

        self.config_manager.update_all_config(new_config)

        config = self.config_manager.get_all_config()
        self.assertEqual(config['regions']['hat']['x1'], 1)
        self.assertEqual(config['default']['score_range'], [0, 10])
        self.assertEqual(config['физика']['10']['score_range'], [0, 5])

    def test_update_all_config_validation(self):
        """Тест валидации при полном обновлении конфига"""
        invalid_config = {
            # Нет regions
            'default': {'score_range': [0, 5]}
        }

        with self.assertRaises(ValueError):
            self.config_manager.update_all_config(invalid_config)

        invalid_config2 = {
            'regions': {},
            # Нет default.score_range
            'default': {}
        }

        with self.assertRaises(ValueError):
            self.config_manager.update_all_config(invalid_config2)

    def test_persistence(self):
        """Тест сохранения конфига в файл"""
        # Меняем конфиг
        self.config_manager.set_score_range('история', '6', (0, 2))

        # Создаем новый ConfigManager с тем же путем
        new_config_manager = ConfigManager(self.config_path)

        # Проверяем что изменения сохранились
        score_range = new_config_manager.get_score_range('история', '6')
        self.assertEqual(score_range, (0, 2))

    def test_load_corrupted_config(self):
        """Тест загрузки поврежденного конфига"""
        # Создаем поврежденный JSON
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write('{invalid json')

        # Должен создать дефолтный конфиг
        config_manager = ConfigManager(self.config_path)
        config = config_manager.get_all_config()

        self.assertIn('regions', config)
        self.assertIn('default', config)


if __name__ == '__main__':
    unittest.main()