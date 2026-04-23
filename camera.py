import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/yolov8n_bunga-kamboja-3/weights/best.pt')

COLORS = {
    'Fresh': (0, 200, 80),
    'Layu': (0, 50, 200)
}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Kamera siap digunakan!Tekan tombol 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5, verbose=False)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[class_id]
        color = COLORS.get(label, (200, 200, 200))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(frame, text, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        

    # Hitung jumlah per kelas
    labels = [model.names[int(b.cls[0])] for b in results[0].boxes]
    fresh_count = labels.count('fresh')
    wilted_count = labels.count('wilted')

    # Info panel atas
    cv2.rectangle(frame, (0, 0), (260, 60), (30, 30, 30), -1)
    cv2.putText(frame, f"Fresh: {fresh_count}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,200,80), 2)
    cv2.putText(frame, f"Layu:  {wilted_count}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,50,220), 2)

    cv2.imshow("Detektor Bunga Kamboja", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
