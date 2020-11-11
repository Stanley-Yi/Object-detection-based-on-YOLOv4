import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import time
import cv2

LABELS = open("coco.names").read().strip().split("\n")
np.random.seed(666)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
# Import the YOLO configuration and weights files and load the network
net = cv2.dnn_DetectionModel('yolov4.cfg', 'yolov4.weights')
# Get the unconnected YOLO output layer
layer = net.getUnconnectedOutLayersNames()
def yolo(img):
    image = img
    # Get picture size
    (H, W) = image.shape[:2]
    # Construct a BLOB from the input image, then perform the forward of the YOLO object detector, giving us the bounding box and the associated probability
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    # Pass it forward, get information
    layerOutputs = net.forward(layer)
    # determine the detection time
    end = time.time()
    print("YOLO took {:.6f} seconds".format(end - start))

    boxes = []
    confidences = []
    classIDs = []

    # Loop out each output layer
    for output in layerOutputs:
        # Loop to extract each box
        for detection in output:
            # Extract the class ID and confidence of the current target
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # The weak prediction is filtered by ensuring that the detection probability is greater than the minimum probability
            if confidence > 0.5:
                # Scale the bounding box coordinates relative to the size of the image. What YOLO returns is the center (x, y) coordinates of the bounding box,
                # This is followed by the width and height of the bounding box
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # Converts the top left corner coordinates of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # Updates the list of bounding box coordinates, confidence, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # Determine a unique bounding box
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    # Make sure that at least one box exists for each object
    if len(idxs) > 0:
        # Loop to draw the saved bounding box
        for i in idxs.flatten():
            # Extract the coordinates and widths
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # Draw bounding box and labels
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1, lineType=cv2.LINE_AA)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, lineType=cv2.LINE_AA)

    return image
