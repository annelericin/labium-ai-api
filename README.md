# Labium Sınıflandırma API

Bu proje, labiominor, labiomajor ve klitoris bölgelerinin sınıflandırmasını yapabilen çoklu çıkışlı bir derin öğrenme modelini Flask API olarak sunar.

## Kullanım

1. Gerekli paketleri yükleyin:
```
pip install -r requirements.txt
```

2. Uygulamayı başlatın:
```
python app.py
```

3. `POST /predict` endpoint’ine bir görsel göndererek tahmin alın.

## Not
İlk çalıştırmada model dosyası otomatik olarak Google Drive’dan indirilecektir.
