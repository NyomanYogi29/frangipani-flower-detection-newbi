from ultralytics import YOLO

model = YOLO('runs/detect/yolov8n_bunga-kamboja/weights/best.pt')
metrics = model.val()

print(f"mAP50: {metrics.box.map50:.2f}")
print(f"mAP50-95: {metrics.box.map:.2f}")
print(f"Precision: {metrics.box.mp:.2f}")
print(f"Recall: {metrics.box.mr:.2f}")
