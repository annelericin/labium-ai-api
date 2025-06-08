from flask import Flask, jsonify
import gdown
import os
import tensorflow as tf

app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
