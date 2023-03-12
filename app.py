import cv2
import torch
import sys
import os
sys.path.append('yolov5')
from models.experimental import attempt_load
from utils.general import non_max_suppression, xyxy2xywh
from utils.torch_utils import select_device

# Load YOLOv5 model
weights = 'yolov5s.pt'
device = select_device('')
model = attempt_load(weights, device)
model.eval()

# Set threshold for object detection
conf_thresh = 0.3

# Load class names for dataset
class_names_file = os.path.join('yolov5', 'data', 'coco.names')
with open(class_names_file, 'r') as f:
    class_names = f.read().splitlines()

# Open video stream or image file
cap = cv2.VideoCapture(0) # or replace 0 with the path to your image file

while True:
    # Read a frame from the video stream or image file
    ret, frame = cap.read()

    # Resize the image to match the expected input size of the model
    img = cv2.resize(frame, (640, 640))

    # Convert the image to a tensor
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    # Run inference on the image
    with torch.no_grad():
        pred = model(img, augment=False)[0]

    # Apply non-maximum suppression to remove overlapping bounding boxes
    pred = non_max_suppression(pred, conf_thresh, 0.4)

    # Display the bounding boxes and class labels
    if pred is not None:
        for det in pred[0]:
            if det is not None and det[-1] > conf_thresh:
                x1, y1, x2, y2, conf, cls = det
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cls_name = class_names[int(cls)]
                label = f'{cls_name}: {conf:.2f}'
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow('frame', frame)

    # Wait for a key press and check if 'q' was pressed to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream or image file and close all windows
cap.release()
cv2.destroyAllWindows()
