from flask import Flask, jsonify, request
from flask_cors import CORS
import gdown
import os
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

MODEL_PATH = "multi_output_model.h5"
DRIVE_FILE_ID = "1iONcsu85I7NHnAGkfmb2hEh51B3KyzkK"

def download_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        print("📥 Model indiriliyor...")
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Model indirildi.")

download_model_from_drive()
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/")
def home():
    return jsonify({"message": "API aktif."})

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Görsel yüklenmedi."}), 400

    image_file = request.files['image']
    image = Image.open(image_file).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)[0]
    prediction_labels = {
        "Labiominor": str(np.argmax(predictions[:3])),
        "Labiomajor": str(np.argmax(predictions[3:6])),
        "Klitoris": str(np.argmax(predictions[6:9]))
    }

    return jsonify({"tahmin": prediction_labels})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
