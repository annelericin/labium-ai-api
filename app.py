from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import gdown
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model/multi_output_model.h5"
DRIVE_FILE_ID = "1iONcsu85I7NHnAGkfmb2hEh51B3KyzkK"

# 🔽 Modeli Drive'dan indir
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# 🔄 Modeli yükle
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model yüklendi.")
        return model
    except Exception as e:
        print("❌ Model yükleme hatası:", e)
        return None

download_model()
model = load_model()

# 🔍 Görseli işleyip tahmin et
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model yüklenemedi'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'Görsel bulunamadı'}), 400

    image = request.files['image'].read()
    try:
        processed = preprocess_image(image)
        preds = model.predict(processed)
        response = {
            'labiominor': preds[0].tolist(),
            'labiomajor': preds[1].tolist(),
            'klitoris': preds[2].tolist()
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': f'Tahmin sırasında hata: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
