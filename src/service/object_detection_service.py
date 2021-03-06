import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("C:/Users/aritz/OneDrive/Escritorio/MUUUH/3. maila/2/PBL/IA/Image-classification-REST-API/config/yolov3.weights",
                    "C:/Users/aritz/OneDrive/Escritorio/MUUUH/3. maila/2/PBL/IA/Image-classification-REST-API/config/yolov3.cfg")
classes = []
with open("C:/Users/aritz/OneDrive/Escritorio/MUUUH/3. maila/2/PBL/IA/Image-classification-REST-API/config/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]


def get_image_object(img):

    # Loading image
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(
        img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    labels = []
    imgsOut = []
    img_copy = img.copy()

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            labels.append(label)
            imgsOut.append(img_copy[y:y+h+10, x:x+w+10])

    return imgsOut, img, labels
