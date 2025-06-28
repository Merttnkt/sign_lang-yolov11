import torch
import cv2
from ultralytics import YOLO

# CUDA'nın mevcut olup olmadığını kontrol edin
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Modeli yükleyin
model = YOLO("sign_model.pt").to(device)

# Kamerayı açın
cap = cv2.VideoCapture(0)

# Tahmin sonuçlarını yazdırmak için dosya oluştur
output_file = open("detection_results.txt", "a")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera görüntüsü alınamadı.")
            break

        # Modeli kullanarak tahmin yapın
        results = model(frame)

        # Çerçeveye tahmin ekleme
        annotated_frame = frame.copy()

        if results[0].boxes:  # Eğer tahminler varsa
            for box in results[0].boxes:
                # Kutunun koordinatlarını ve sınıf bilgisini alın
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Kutunun sol üst ve sağ alt köşeleri
                cls_id = int(box.cls[0])  # Sınıf kimliği
                class_name = model.names[cls_id]  # Sınıf ismi

                # Konsola yazdır
                print(f"Sınıf: {class_name}, Koordinatlar: ({x1}, {y1}), ({x2}, {y2})")

                # Dosyaya yazdır
                output_file.write(f"Sınıf: {class_name}, Koordinatlar: ({x1}, {y1}), ({x2}, {y2})\n")

                # Çerçeveyi çizin
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Sınıf ismini kutunun üstüne yazın
                cv2.putText(annotated_frame, class_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Sonuçları ekranda gösterin
        cv2.imshow("Hand Gesture Detection", annotated_frame)

        # 'q' tuşuna basıldığında çıkış yap
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Kaynakları serbest bırak
    cap.release()
    cv2.destroyAllWindows()
    output_file.close()  # Dosyayı kapat
