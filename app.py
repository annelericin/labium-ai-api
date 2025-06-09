from flask import Flask, jsonify, request
from flask_cors import CORS
import gdown
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import io

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
    if 'file' not in request.files:
        return jsonify({"error": "Dosya yüklenmedi."}), 400

    file = request.files['file']
    img = Image.open(file.stream).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    labels = ['label1', 'label2', 'label3']  # ← burayı kendi sınıf etiketlerinle değiştir

    result = {label: float(pred) for label, pred in zip(labels, predictions)}
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
