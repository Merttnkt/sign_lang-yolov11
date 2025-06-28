from flask import Flask, request, jsonify
import torch
import cv2
from ultralytics import YOLO
import numpy as np

# Flask uygulamasını başlat
app = Flask(__name__)

# CUDA'nın mevcut olup olmadığını kontrol et
device = "cuda" if torch.cuda.is_available() else "cpu"

# Modeli yükle
model = YOLO("sign_model.pt").to(device)

@app.route("/detect", methods=["POST"])
def detect():
    # Gelen istekte bir dosya var mı kontrol edin
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file part in the request"}), 400

    # Resim dosyasını alın
    image = request.files['image']

    # Dosyanın seçilip seçilmediğini kontrol edin
    if image.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    try:
        # Resmi OpenCV ile yükleyin
        np_img = np.frombuffer(image.read(), np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Modeli kullanarak tahmin yapın
        results = model(frame)

        # Tahmin sonuçlarını işleyin
        detections = []
        if results[0].boxes:  # Eğer tahminler varsa
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinatlar
                cls_id = int(box.cls[0])  # Sınıf kimliği
                class_name = model.names[cls_id]  # Sınıf adı
                detections.append({
                    "class": class_name,
                    "box": [x1, y1, x2, y2]
                })

        # Başarı yanıtı dön
        return jsonify({"status": "success", "detections": detections}), 200

    except Exception as e:
        # Hata durumunda mesaj dön
        return jsonify({"status": "error", "message": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
