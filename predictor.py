# predictor.py içine ekle
import numpy as np

def predict_classes(model, image_array):
    predictions = model.predict(image_array)
    results = {}
    output_labels = ['labia_minora', 'labia_majora', 'clitoris']  # kendi sınıflarına göre düzenle
    for i, label in enumerate(output_labels):
        predicted_class = np.argmax(predictions[i])
        results[label] = predicted_class
    return results
