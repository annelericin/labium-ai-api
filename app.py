import os
import gdown
from tensorflow.keras.models import load_model
from predictor import predict_classes  # Tahmin fonksiyonunu içeren dosya

MODEL_PATH = "multi_output_model.h5"
DRIVE_FILE_ID = "1iONcsu85I7NHnAGkfmb2hEh51B3KyzkK"

# 📥 Modeli indir
def download_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        print("📥 Model indiriliyor...")
        gdown.download(id=DRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
        print("✅ Model indirildi.")

# 🧠 Modeli yükle
download_model_from_drive()
model = load_model(MODEL_PATH)

# 📦 Sınıf isimleri
minor_classes = ['asimetrik', 'hipertrofik', 'hipoplazik', 'normal']
major_classes = ['asimetrik', 'hipertrofik', 'hipoplazik', 'normal', 'yağlı']
klitoris_classes = ['bifid', 'hipertrofik', 'normal', 'yok', 'küçük']

# 🌐 Flask API
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return 'Labium AI API Çalışıyor!'

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Görsel yüklenmedi.'}), 400

    img_file = request.files['image']
    img = Image.open(img_file).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)

    result = {
        'labiominor': minor_classes[np.argmax(preds[0])],
        'labiomajor': major_classes[np.argmax(preds[1])],
        'klitoris': klitoris_classes[np.argmax(preds[2])]
    }

    return jsonify(result)

if __name__ == '__main__':
  import os

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)

