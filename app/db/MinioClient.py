import os
from minio import Minio
from minio.error import S3Error
from io import BytesIO


class MinioClient:
    def __init__(self):
        self.client = None
        self.bucket = os.getenv("MINIO_BUCKET")
        self.init()

    def get_client(self):
        if self.client is not None:
            return self.client

        endpoint = os.getenv("MINIO_ENDPOINT")
        access_key = os.getenv("MINIO_ACCESS_KEY")
        secret_key = os.getenv("MINIO_SECRET_KEY")

        if not access_key or not secret_key:
            raise RuntimeError("MINIO_ACCESS_KEY или MINIO_SECRET_KEY не заданы")

        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            # todo поддержать secure
            secure=False,
        )
        return self.client

    def init_bucket(self):
        try:
            client = self.get_client()
            if not client.bucket_exists(self.bucket):
                client.make_bucket(self.bucket)
        except S3Error as e:
            print(f"Ошибка при инициализации бакета {self.bucket}: {e}")

    def upload_bytes(self, data, object_name, content_type):
        client = self.get_client()
        size = len(data)

        client.put_object(
            bucket_name=self.bucket,
            object_name=object_name,
            data=BytesIO(data),
            length=size,
            content_type=content_type,
        )

    def init(self):
        self.init_bucket()
