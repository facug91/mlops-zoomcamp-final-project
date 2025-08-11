import io
import uuid
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from services.prediction_logger import PredictionLogger


@pytest.fixture
def dummy_image():
    """
    Fixture: Creates a 100x100 red image for testing.
    Avoids using external files, ensures reproducible test input.
    """
    return Image.new("RGB", (100, 100), color="red")


@patch("psycopg2.connect")
def test_prepare_db_executes_create_table(mock_connect):
    """
    Test that PredictionLogger.prepare_db executes the CREATE TABLE statement.

    This test mocks psycopg2.connect to ensure that:
    - No real database is accessed.
    - The SQL for creating the predictions table is executed exactly once.
    """
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    # Instantiating PredictionLogger automatically calls prepare_db()
    logger = PredictionLogger()

    # Ensure CREATE TABLE was called exactly once
    assert mock_cursor.execute.call_count == 1
    args, kwargs = mock_cursor.execute.call_args
    assert "CREATE TABLE IF NOT EXISTS predictions" in args[0]


@patch("psycopg2.connect")
def test_save_to_db_inserts_correct_values(mock_connect):
    """
    Test that PredictionLogger.save_to_db inserts the correct values.

    This test verifies:
    - The SQL starts with INSERT INTO predictions.
    - The image_path is included in the parameters.
    - The exec_time_ms is an integer in milliseconds.
    """
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    logger = PredictionLogger()
    assert mock_cursor.execute.call_count == 1
    predictions = [("Apple", 0.8), ("Banana", 0.05), ("Grape", 0.05), ("Banana", 0.05), ("Mango", 0.05)]
    duration = 1.234
    image_path = "s3://bucket/predictions/test.jpg"

    logger.save_to_db(predictions, duration, image_path)

    # Ensure an INSERT was executed
    assert mock_cursor.execute.call_count == 2
    sql, params = mock_cursor.execute.call_args[0]
    assert sql.startswith("INSERT INTO predictions")
    assert image_path in params
    
    # Verify exec_time_ms is an integer and matches milliseconds conversion
    exec_time_ms = params[1]
    assert isinstance(exec_time_ms, int)
    assert exec_time_ms == int(duration * 1000)


@patch("psycopg2.connect")
def test_save_to_db_contains_expected_fruit_columns(mock_connect):
    """
    Test that the columns inserted into the database match the fruit labels from predictions.

    This ensures that the generated SQL column list includes all provided fruit labels in lowercase.
    """
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    logger = PredictionLogger()
    predictions = [("Apple", 0.7), ("Mango", 0.15), ("Strawberry", 0.05), ("Banana", 0.05), ("Mango", 0.05)]
    duration = 0.5
    image_path = "s3://bucket/predictions/test2.jpg"

    logger.save_to_db(predictions, duration, image_path)

    # Get the SQL string used in the INSERT
    sql, params = mock_cursor.execute.call_args[0]

    # Extract the part of the SQL that lists the columns
    start = sql.find("(") + 1
    end = sql.find(")")
    columns_str = sql[start:end]
    inserted_columns = [col.strip() for col in columns_str.split(",")]

    # Expected fruit columns (lowercase)
    expected_fruits = [label.lower() for label, _ in predictions]

    # Ensure all expected fruit columns are present
    for fruit in expected_fruits:
        assert fruit in inserted_columns


@patch("boto3.client")
@patch("psycopg2.connect")
def test_upload_image_to_s3_returns_expected_path(mock_connect, mock_boto_client, dummy_image):
    """
    Test that PredictionLogger.upload_image_to_s3 uploads the image
    and returns the correct S3 path.

    This test verifies:
    - upload_fileobj is called exactly once.
    - The bucket and key match the expected values.
    - The returned path matches the S3 bucket/key format.
    """
    mock_s3 = MagicMock()
    mock_boto_client.return_value = mock_s3

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    
    fake_run_id = uuid.uuid4().hex
    logger = PredictionLogger()
    logger.s3_config = {
        "endpoint_url": "http://localhost:9000",
        "access_key": "fake",
        "secret_key": "fake",
        "region": "us-east-1",
        "bucket": "test-bucket"
    }

    result_path = logger.upload_image_to_s3(dummy_image, run_id=fake_run_id)

    # Ensure upload_fileobj was called exactly once
    assert mock_s3.upload_fileobj.call_count == 1
    args, kwargs = mock_s3.upload_fileobj.call_args
    uploaded_buffer = args[0]
    uploaded_bucket = args[1]
    uploaded_key = args[2]

    # The buffer should be an in-memory BytesIO object
    assert isinstance(uploaded_buffer, io.BytesIO)
    assert uploaded_bucket == "test-bucket"
    assert uploaded_key == f"predictions/{fake_run_id}.jpg"

    # Returned path should match expected S3 URI format
    assert result_path == f"s3://test-bucket/predictions/{fake_run_id}.jpg"
