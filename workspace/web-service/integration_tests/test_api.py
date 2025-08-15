import io
import os

import requests
from PIL import Image

BASE_URL = os.getenv("WEB_SERVICE_URL", "http://localhost:8080")
VALID_LABELS = {"apple", "banana", "mango", "strawberry", "grape"}


def create_test_image(color=(255, 0, 0)):
    """Crea una imagen RGB simple en memoria."""
    image = Image.new("RGB", (64, 64), color)
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


def test_predict_success():
    img_bytes = create_test_image()
    files = {"image": ("test.jpg", img_bytes, "image/jpeg")}
    response = requests.post(f"{BASE_URL}/predict", files=files)

    assert response.status_code == 200
    data = response.json()

    assert "predictions" in data
    preds = data["predictions"]

    # Verifica que el label sea válido
    assert preds["label"] in VALID_LABELS

    # Verifica que la probabilidad esté entre 0 y 1
    assert isinstance(preds["prob"], (float, int))
    assert 0.0 <= preds["prob"] <= 1.0


def test_predict_no_image():
    response = requests.post(f"{BASE_URL}/predict", files={})
    assert response.status_code == 400
    data = response.json()
    assert "error" in data


def test_predict_invalid_file():
    fake_file = io.BytesIO(b"%PDF-1.4 fake pdf content here")
    files = {"image": ("fake.pdf", fake_file, "application/pdf")}
    response = requests.post(f"{BASE_URL}/predict", files=files)

    assert response.status_code == 500  # o el código que definas
    data = response.json()
    assert "error" in data
