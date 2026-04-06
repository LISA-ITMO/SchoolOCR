import json
from numpy import ndarray
from app.db.Db import Db
from app.db.MinioClient import MinioClient
import cv2
from time import time


class RecognizeLogger:
    def __init__(self):
        self.minioClient = MinioClient()
        self.dbClient = Db()

    def convert_image_to_bytes(self, np_image: ndarray) -> bytes:
        success, encoded_img = cv2.imencode(".jpg", np_image)
        if not success:
            raise ValueError("Ошибка в преобразовании картинки")
        return encoded_img.tobytes()

    def log(
        self,
        initial_image: ndarray,
        preprocessed_image: ndarray,
        recognation_result: float,
        recognation_accuracy: float,
        recognation_detail: dict[str, float],
    ):
        initial_image_bytes = self.convert_image_to_bytes(initial_image)
        preprocessed_image_bytes = self.convert_image_to_bytes(preprocessed_image)

        common_name = f"{time()}_{recognation_result}"
        initial_image_name = f"{common_name}.jpg"
        preprocessed_image_name = f"{common_name}_preprocessed.jpg"

        self.minioClient.upload_bytes(
            data=initial_image_bytes,
            object_name=initial_image_name,
            content_type="image/jpeg",
        )
        self.minioClient.upload_bytes(
            data=preprocessed_image_bytes,
            object_name=preprocessed_image_name,
            content_type="image/jpeg",
        )

        initial_image_url = self.minioClient.get_public_url(initial_image_name)
        preprocessed_image_url = self.minioClient.get_public_url(
            preprocessed_image_name
        )

        insert_params = (
            initial_image_url,
            preprocessed_image_url,
            recognation_result,
            recognation_accuracy,
            json.dumps(recognation_detail),
        )

        try:
            self.dbClient.query(
                """
                    INSERT INTO recognize_logs (cell_img, cell_img_after_preprocessing, recognize_result, recognation_accuracy, recognation_detail)
                    values (%s, %s, %s, %s, %s)
                """,
                insert_params,
            )
        except Exception as e:
            print(e)
