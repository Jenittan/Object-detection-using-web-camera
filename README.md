# Object-detection-using-web-camera
A simple project to perform real-time object detection using a web camera. The system captures live video feed, processes each frame, and detects objects using a pre-trained model.
## Features:
- Real-time object detection using a web camera
- Highlights objects with bounding boxes and labels
- Works with pre-trained models (YOLO / MobileNet / Haar Cascade)
- Easy to customize for specific object categories


## Technologies Used:
- Python
- OpenCV – for webcam access and image processing
- Numpy – for numerical operations
- TensorFlow / PyTorch – for deep learning models (if used)
- YOLO / MobileNet-SSD / Haar Cascade – object detection models
## How to Use:
1. Clone this repository.
2. Install dependencies.
3. Run the project.
4. The webcam feed will open, and detected objects will be displayed with bounding boxes.

## Applications:
- Surveillance & Security – detecting people or suspicious objects
- Traffic Monitoring – vehicles and pedestrian detection
- Retail Analytics – customer tracking and insights
- Robotics – enabling robots to “see” their environment
- Research & Education – learning AI and computer vision

## program
```

import cv2
import numpy as np

# Load YOLOv4 network
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Set up video capture for webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # Prepare the image for YOLOv4
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get YOLO output
    outputs = net.forward(output_layers)
    
    # Initialize lists to store detected boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate top-left corner of the box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the image
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            color = (0, 255, 0)  # Green color for bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the image with detected objects
    cv2.imshow("YOLOv4 Real-Time Object Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
 
```
## output:
<img width="789" height="569" alt="Screenshot 2025-09-27 132106" src="https://github.com/user-attachments/assets/42db99b2-bfb1-4008-a811-ac7a5be7c531" />


Feel free to fork, contribute, or customize this project for your creative needs!
