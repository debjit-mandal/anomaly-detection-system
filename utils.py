import cv2
import numpy as np
import requests

def load_yolo_model(cfg_path, weights_path):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    layer_names = net.getLayerNames()
    try:
        unconnected_layers = net.getUnconnectedOutLayers()
        if len(unconnected_layers.shape) == 2:
            output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]
        else:
            output_layers = [layer_names[i - 1] for i in unconnected_layers]
    except TypeError:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def detect_objects(net, output_layers, frame, height, width):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    objects = []

    with open("data/class_names.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            label = str(classes[class_ids[i]])
            objects.append({"box": box, "label": label})

    return objects

def send_alert():
    url = "https://your-webhook-url"
    data = {"message": "Anomaly detected!"}
    response = requests.post(url, json=data)
    print(response.status_code)
