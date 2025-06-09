from flask import Flask, jsonify, request
from flask_cors import CORS
import gdown
import os
import tensorflow as tf
import numpy as np
from PIL import Image

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
        return jsonify({"error": "Resim dosyası gerekli."}), 400
    
    file = request.files['image']
    try:
        image = Image.open(file.stream).resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        return jsonify({"tahmin": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
