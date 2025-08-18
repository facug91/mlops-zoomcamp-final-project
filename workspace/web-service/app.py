import io
import logging
import time
import uuid

from flask import Flask, jsonify, render_template, request
from PIL import Image
from services.model import ModelService
from services.prediction_logger import PredictionLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


# Initialize Flask app and model
app = Flask("fruits-classifier-web-service")
model_service = ModelService()
prediction_logger = PredictionLogger()


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image_file = request.files["image"]

        start_time = time.perf_counter()
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        predictions = model_service.predict(image)
        end_time = time.perf_counter()
        exec_time_sec = end_time - start_time

        image_path = prediction_logger.upload_image_to_s3(image, uuid.uuid4().hex)
        width, height = image.size
        prediction_logger.save_to_db(predictions, exec_time_sec, image_path, width, height)

        label, prob = predictions[0]
        result = {"label": label, "prob": prob}
        return jsonify({"predictions": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return render_template("index.html")


# flask run --host 0.0.0.0 --port 8080 --debug
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
