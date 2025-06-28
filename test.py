import os
import torch
from ultralytics import YOLO

# CUDA kullanımı için bellek temizleme
torch.cuda.empty_cache()


model = YOLO("sign_model.pt") 

test = './datasets/data.yaml'

# CUDA kullanılabilir mi kontrol edin
print("CUDA Available:", torch.cuda.is_available())


results = model.val(data=test, batch=10, imgsz=(640, 640))


log_dir = './runs/test/yolo_sign_test'  
os.makedirs(log_dir, exist_ok=True)

# Metrikleri ekrana yazdırın
print("Test Sonuçları:")
print(f"Precision: {results['metrics/precision']}")
print(f"Recall: {results['metrics/recall']}")
print(f"mAP@50: {results['metrics/mAP50']}")
print(f"mAP@50-95: {results['metrics/mAP50-95']}")


results.save_dir = log_dir 
print(f"Sonuçlar {log_dir} dizinine kaydedildi.")

#cümle kurma 