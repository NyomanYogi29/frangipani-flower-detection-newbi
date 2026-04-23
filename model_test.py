from ultralytics import YOLO

model = YOLO('runs/detect/yolov8n_bunga-kamboja/weights/best.pt')
result = model('datasets/images/train/0fc975ef-1776914757940.jpg', save=True, conf=0.5)