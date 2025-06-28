import os
import torch
from ultralytics import YOLO

train_data = './datasets/data.yaml'

model = YOLO('yolov10n.pt')

print(torch.cuda.is_available())


log_dir = './runs/train/yolo_'

model.train(
    data=train_data,
    epochs=150,
    batch=10,
    imgsz=(640, 640),
    project=log_dir,
    name='yolo_sign',
    save=True,
    plots=True,
    device="cuda"
)



model.save('sign_model.pt')

print("Training complete. Results saved to:", log_dir)


