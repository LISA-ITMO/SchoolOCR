import json
import os
from typing import Dict, Tuple, Optional, Any, List

CONFIG_PATH = "config_new.json"

class ConfigManager:
    def __init__(self, config_path: str = None):
        if config_path is None:
            current_dir = os.path.dirname(__file__)
            self.config_path = os.path.join(current_dir, CONFIG_PATH)
        else:
            self.config_path = config_path

        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    print(f"Конфиг загружен из {self.config_path}")
                    return config
            else:
                print(f"Файл конфига не найден, создается дефолтный: {self.config_path}")
                return self._create_default_config()

        except Exception as e:
            print(f"Ошибка загрузки конфига: {e}, создается дефолтный")
            return self._create_default_config()

    def _create_default_config(self) -> Dict[str, Any]:
        default_config = {
            "regions": {
                "hat": {
                    "x1": 0,
                    "y1": 0,
                    "x2": 1489,
                    "y2": 400
                },
                "code": {
                    "x1": 1489,
                    "y1": 0,
                    "x2": 2400,
                    "y2": 400
                },
                "hat_reserve": {
                    "x1": 0,
                    "y1": 0,
                    "x2": 1600,
                    "y2": 400
                }
            },
            "default": {
                "score_range": [0, 9],
                "recognition": {
                    "use_llm": False,
                    "confidence_threshold": 0.6,
                    "llm_trigger_threshold": 0.6
                }
            },
            "llm": {
                "api_url": "http://localhost:11434/api/chat",
                "model": "qwen2.5vl:7b",
                "prompt": (
                    "Ты система извлечения данных из экзаменационного бланка.\n\n"
                    "Нужно полностью распознать документ и вернуть СТРОГО JSON.\n\n"
                    "Запрещено:\n"
                    "- писать пояснения\n"
                    "- писать текст до или после JSON\n"
                    "- использовать markdown\n"
                    "- добавлять ```json```\n\n"
                    "Если значение невозможно прочитать — null.\n"
                    "Не выдумывай данные.\n\n"
                    "Формат ответа:\n\n"
                    "{\n"
                    '  "subject": string|null,\n'
                    '  "grade": string|null,\n'
                    '  "variant": string|null,\n'
                    '  "participant_code": string|null,\n'
                    '  "total_score": integer|null,\n'
                    '  "scores": {\n'
                    '    "<номер задания>": integer|string|null\n'
                    "  },\n"
                    '  "errors": null,\n'
                    '  "warnings": null\n'
                    "}\n\n"
                    "ПРАВИЛА ДЛЯ scores:\n"
                    "- это словарь (НЕ массив)\n"
                    '- ключи — строки: "1", "2", "3", ...\n'
                    "- количество заданий НЕ фиксировано\n"
                    "- добавь ВСЕ строки таблицы, которые видишь\n"
                    "- если номера заданий напечатаны — используй их\n"
                    "- если номеров нет — нумеруй сверху вниз начиная с 1\n"
                    "- не придумывай отсутствующие задания\n\n"
                    "Допустимые значения в таблице:\n"
                    "- целое число\n"
                    '- "X"\n'
                    '- "-"\n'
                    "- null\n\n"
                    "Правила распознавания:\n"
                    "- код участника — РОВНО 5 цифр (ведущие нули сохраняй)\n"
                    "- рукописные цифры переписывай буквально\n"
                    "- символ X — латинская заглавная буква\n"
                    '- X и "-" НЕ заменять на числа\n'
                    "- если символ неразборчив — null\n"
                    "- суммарный балл — это total_score, если его можно определить\n\n"
                    "Ответ должен быть ТОЛЬКО JSON."
                )
            }
        }

        # Сохраняем дефолтный конфиг
        self._save_config(default_config)
        return default_config

    def _save_config(self, config: Dict[str, Any]):
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"Конфиг сохранен в {self.config_path}")
        except Exception as e:
            print(f"Ошибка сохранения конфига: {e}")

    def get_all_config(self) -> Dict[str, Any]:
        return self._config.copy()

    def get_regions_config(self) -> Dict[str, Any]:
        return self._config.get("regions", {})

    def get_region_coordinates(self, region_name: str) -> Optional[Tuple[int, int, int, int]]:
        regions = self.get_regions_config()
        region = regions.get(region_name)
        if region:
            return (region["x1"], region["y1"], region["x2"], region["y2"])
        return None

    def update_regions_config(self, regions_config: Dict[str, Any]):
        self._config["regions"] = regions_config
        self._save_config(self._config)
        print("Конфигурация регионов обновлена")

    def update_region_coordinates(self, region_name: str, x1: int, y1: int, x2: int, y2: int):
        if "regions" not in self._config:
            self._config["regions"] = {}

        self._config["regions"][region_name] = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }
        self._save_config(self._config)
        print(f"Координаты региона '{region_name}' обновлены: ({x1}, {y1}, {x2}, {y2})")

    def get_subject_config(self, subject: str) -> Optional[Dict[str, Any]]:
        return self._config.get(subject)

    def get_grade_config(self, subject: str, grade: str) -> Optional[Dict[str, Any]]:
        subject_config = self.get_subject_config(subject)
        if subject_config:
            return subject_config.get(grade)
        return None

    def get_score_range(self, subject: str, grade: str) -> Tuple[int, int]:
        grade_config = self.get_grade_config(subject, grade)
        if grade_config and "score_range" in grade_config:
            return tuple(grade_config["score_range"])

        default_range = self._config["default"]["score_range"]
        print(f"Для {subject} {grade} класс не найден конфиг, используется дефолтный диапазон: {default_range}")
        return tuple(default_range)

    def update_all_config(self, new_config: Dict[str, Any]):
        if "regions" not in new_config:
            raise ValueError("Конфиг должен содержать regions")
        if "default" not in new_config or "score_range" not in new_config["default"]:
            raise ValueError("Конфиг должен содержать default.score_range")

        self._config = new_config
        self._save_config(new_config)
        print("Вся конфигурация обновлена")

    def update_subject_config(self, subject: str, subject_config: Dict[str, Any]):
        self._config[subject] = subject_config
        self._save_config(self._config)
        print(f"Конфигурация для предмета '{subject}' обновлена")

    def update_grade_config(self, subject: str, grade: str, grade_config: Dict[str, Any]):
        if subject not in self._config:
            self._config[subject] = {}

        self._config[subject][grade] = grade_config
        self._save_config(self._config)
        print(f"Конфигурация для '{subject} {grade} класс' обновлена")

    def set_score_range(self, subject: str, grade: str, score_range: Tuple[int, int]):
        if subject not in self._config:
            self._config[subject] = {}

        if grade not in self._config[subject]:
            self._config[subject][grade] = {}

        self._config[subject][grade]["score_range"] = list(score_range)
        self._save_config(self._config)
        print(f"Диапазон баллов для '{subject} {grade} класс' установлен: {score_range}")

    def delete_subject(self, subject: str):
        if subject in self._config and subject not in ["regions", "default"]:
            del self._config[subject]
            self._save_config(self._config)
            print(f"Конфигурация предмета '{subject}' удалена")
        else:
            print(f"Предмет '{subject}' не найден или является protected")

    def delete_grade(self, subject: str, grade: str):
        if subject in self._config and grade in self._config[subject]:
            del self._config[subject][grade]
            self._save_config(self._config)
            print(f"Конфигурация для '{subject} {grade} класс' удалена")

            # Если у предмета не осталось классов, удаляем предмет
            if not self._config[subject]:
                del self._config[subject]
                print(f"Предмет '{subject}' удален (не осталось классов)")
        else:
            print(f"Конфигурация для '{subject} {grade} класс' не найдена")

    def list_subjects(self) -> List[str]:
        protected_keys = ["regions", "default"]
        return [subject for subject in self._config.keys() if subject not in protected_keys]

    def list_grades(self, subject: str) -> List[str]:
        if subject in self._config:
            return list(self._config[subject].keys())
        return []

    def get_recognition_config(self) -> Dict[str, Any]:
        """
        Настройки распознавания из default.recognition (одинаковые для всех).
        """
        recog = (self._config.get("default", {}).get("recognition", {}) or {}).copy()

        # safety defaults
        recog.setdefault("use_llm", False)
        recog.setdefault("confidence_threshold", 0.6)
        recog.setdefault("llm_trigger_threshold", recog["confidence_threshold"])

        return recog

    def get_llm_config(self) -> Dict[str, Any]:
        llm_cfg = (self._config.get("default", {}).get("llm", {}) or {}).copy()

        llm_cfg.setdefault("api_url", "http://ollama:11434/api/chat")
        llm_cfg.setdefault("model", "qwen2.5vl:7b")
        llm_cfg.setdefault("prompt", "")

        return llm_cfg

config_manager = ConfigManager()