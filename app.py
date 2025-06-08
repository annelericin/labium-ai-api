import os
import gdown
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

MODEL_PATH = "multi_output_model.h5"
DRIVE_FILE_ID = "1iONcsu85I7NHnAGkfmb2hEh51B3KyzkK"

def download_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        print("📥 Model indiriliyor...")
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Model indirildi.")

download_model_from_drive()
model = load_model(MODEL_PATH)

minor_classes = ['asimetrik', 'hipertrofik', 'hipoplazik', 'normal']
major_classes = ['asimetrik', 'hipertrofik', 'hipoplazik', 'normal', 'yağlı']
klitoris_classes = ['bifid', 'hipertrofik', 'normal', 'yok', 'küçük']

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Görsel bulunamadı."}), 400

    img_file = request.files["image"]
    img_path = "temp.jpg"
    img_file.save(img_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    os.remove(img_path)

    result = {
        "labiominor": minor_classes[np.argmax(predictions[0])],
        "labiomajor": major_classes[np.argmax(predictions[1])],
        "klitoris": klitoris_classes[np.argmax(predictions[2])],
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)