import boto3
import io
import psycopg2
import os
import uuid

from datetime import datetime
from PIL import Image
from typing import List, Tuple


db_config = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

s3_config = {
    "endpoint_url": os.getenv("S3_ENDPOINT_URL"),
    "access_key": os.getenv("AWS_ACCESS_KEY_ID"),
    "secret_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "region": os.getenv("AWS_DEFAULT_REGION"),
    "bucket": os.getenv("S3_BUCKET")
}

class PredictionLogger:
    """
    Handles logging of prediction results to a database and image storage.
    """

    def __init__(self, db_config: dict = db_config, s3_config: dict = s3_config):
        self.db_config = db_config
        self.s3_config = s3_config
        self.prepare_db()

    def prepare_db(self):
        """
        Prepares the database. Create 'metricsdb' if it doesn't exist, and the same for 'predictions' table.
        """
        create_table_statement = """
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                image_path TEXT,
                exec_time_ms INTEGER NOT NULL,
                apple FLOAT,
                banana FLOAT,
                grape FLOAT,
                mango FLOAT,
                strawberry FLOAT
            )
            """
        
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_statement)
                conn.commit()


    def save_to_db(self, predictions: List[Tuple[str, float]], duration: float, image_path: str):
        """
        Save predictions to the database.

        Args:
            predictions (list): List of tuples (label, confidence).
            duration (float): Inference duration in seconds.
            image_path (str): S3 path of the processed image.
        """

        timestamp = datetime.utcnow()
        exec_time_ms = int(duration * 1000.0)
        confidence_dict = {label.lower(): confidence for label, confidence in predictions}

        columns = ["timestamp", "exec_time_ms", "image_path"] + list(confidence_dict.keys())
        values = [timestamp, exec_time_ms, image_path] + list(confidence_dict.values())

        placeholders = ', '.join(['%s'] * len(values))
        columns_str = ', '.join(columns)

        insert_statement = f"INSERT INTO predictions ({columns_str}) VALUES ({placeholders})"

        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_statement, values)
        except Exception as e:
            print(f"Database insert failed: {e}")
            raise


    def upload_image_to_s3(self, image: Image.Image, run_id: str = uuid.uuid4().hex) -> str:
        """
        Uploads the image to an S3 bucket and returns the image path.

        Args:
            image (PIL.Image.Image): The image to upload.
            run_id (str): A unique ID to link this image to a prediction.

        Returns:
            str: The S3 path of the uploaded image.
        """
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)

        s3_key = f"predictions/{run_id}.jpg"

        s3 = boto3.client(
            "s3",
            endpoint_url=self.s3_config["endpoint_url"],
            aws_access_key_id=self.s3_config["access_key"],
            aws_secret_access_key=self.s3_config["secret_key"],
            region_name=self.s3_config["region"]
        )

        s3.upload_fileobj(buffer, self.s3_config["bucket"], s3_key, ExtraArgs={"ContentType": "image/jpeg"})

        return f"s3://{self.s3_config['bucket']}/{s3_key}"

