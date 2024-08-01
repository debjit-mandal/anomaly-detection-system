import cv2
import numpy as np
import torch
from anomaly_detector import AnomalyDetector, is_anomaly
from utils import load_yolo_model, detect_objects, send_alert

net, output_layers = load_yolo_model("models/yolov3.cfg", "models/yolov3.weights")

anomaly_model = AnomalyDetector()
anomaly_model.load_state_dict(torch.load("anomaly_model.pth"))
anomaly_model.eval()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    objects = detect_objects(net, output_layers, frame, height, width)

    if is_anomaly(frame, anomaly_model):
        send_alert()

    for obj in objects:
        x, y, w, h = obj["box"]
        label = obj["label"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
