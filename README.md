
# Real-Time Anomaly Detection

![image](https://github.com/user-attachments/assets/6da28c57-95fa-4875-8462-2091ce0a5fa5)


This project implements a real-time anomaly detection system using OpenCV, YOLO (You Only Look Once) for object detection, and a custom anomaly detection model built with PyTorch. The system uses a webcam to monitor and detect anomalies such as unauthorized access, violence, or unusual activities in real-time.

## Features

- Real-time object detection using YOLOv3
- Custom anomaly detection model
- Real-time alert system using webhooks
- Integration with a mobile app for alerts (optional)
- Self-learning mechanism to improve anomaly detection over time

## Technologies Used

- OpenCV
- YOLO (You Only Look Once)
- PyTorch
- Python
- Webhooks (for real-time alerts)

## Directory Structure

```
anomaly_detection/
│
├── models/
│   ├── yolov3.cfg
│   └── yolov3.weights
│
├── data/
│   ├── class_names.txt
│   ├── coco/
│   │   ├── train2017/
│   │   └── val2017/
│
├── main.py
├── anomaly_detector.py
├── train_anomaly_detector.py
├── create_placeholder_model.py
├── requirements.txt
├── utils.py
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.x
- pip (Python package installer)
- OpenCV
- PyTorch
- Git (optional, for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/debjit-mandal/anomaly-detection-system.git
cd anomaly-detection-system
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download YOLO Configuration and Weights

```bash
# Create directories
mkdir -p models

# Download YOLOv3 Configuration File
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -P models/

# Download YOLOv3 Weights File
wget https://pjreddie.com/media/files/yolov3.weights -P models/
```

### Step 4: Download and Extract the COCO Dataset

```bash
# Create directories
mkdir -p data/coco

# Download COCO 2017 Train Images
wget http://images.cocodataset.org/zips/train2017.zip -P data/coco/

# Download COCO 2017 Validation Images
wget http://images.cocodataset.org/zips/val2017.zip -P data/coco/

# Optional: Download COCO 2017 Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P data/coco/

# Extract Train Images
unzip data/coco/train2017.zip -d data/coco/

# Extract Validation Images
unzip data/coco/val2017.zip -d data/coco/

# Optional: Extract Annotations
unzip data/coco/annotations_trainval2017.zip -d data/coco/
```

### Step 5: Train the Anomaly Detection Model

```bash
python train_anomaly_detector.py
```

### Step 6: Run the Real-Time Anomaly Detection System

```bash
python main.py
```

## Usage

- The script will open your webcam and start detecting objects and anomalies in real-time.
- Detected objects will be displayed with bounding boxes and labels.
- If an anomaly is detected, an alert will be sent via a webhook.

## Customization

### Adding New Object Classes

- Edit the `data/class_names.txt` file to include new object classes.

### Adjusting Detection Thresholds

- Modify the confidence threshold in the `detect_objects` function in `utils.py`.

### Using a Different Webcam

- Change the webcam index in `main.py` if you have multiple webcams:

```python
cap = cv2.VideoCapture(0)  # Change 0 to the appropriate index
```

## Contributing

Contributions are welcome! Please fork this repository and submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLO: You Only Look Once
- COCO Dataset
- PyTorch
