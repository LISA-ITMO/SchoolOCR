import psycopg2
import os


class Db:
    def __init__(self):
        self.connection = None
        self.init()

    def get_connection(self):
        if self.connection is not None:
            return self.connection

        self.connection = psycopg2.connect(
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
        )

        return self.connection

    def init_recognize_logs_table(self):
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS recognize_logs 
                (id serial PRIMARY KEY, cell_img text, cell_img_after_preprocessing text, 
                recognize_result text, recognation_accuracy text, recognation_detail jsonb);"""
            )
            connection.commit()
            cursor.close()
        except Exception as e:
            print(f"Ошибка при инициализации таблицы recognize_logs: {e}")

    def query(self, queryString, params):
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            cursor.execute(queryString, params)
            connection.commit()
            cursor.close()
        except Exception:
            print("Ошибка при исполнении запроса в БД")

    def init(self):
        self.init_recognize_logs_table()
