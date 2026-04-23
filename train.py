from ultralytics import YOLO
model = YOLO('yolov8n.pt')

results = model.train(
    data='bunga-kamboja.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='yolov8n_bunga-kamboja',
    patience=20,
    augment=True
)

print("Training selesai!")