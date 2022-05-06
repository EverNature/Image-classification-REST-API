from fastapi import FastAPI, File, UploadFile
import base64
import uvicorn
import numpy as np
import tensorflow as tf
from datetime import datetime
from opencv import get_image_object, cv2

app = FastAPI()

MODEL = tf.keras.models.load_model("C:/Users/aritz/OneDrive/Escritorio/MUUUH/3. maila/2/PBL/IA/saved_models/vgg16_weights_full_model")

LABEL_NAMES = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

CLASS_NAMES = ['ABBOTTS BABBLER',
 'ABBOTTS BOOBY',
 'AFRICAN FIREFINCH',
 'AFRICAN OYSTER CATCHER',
 'BEARDED BARBET',
 'CASPIAN TERN',
 'CEDAR WAXWING',
 'GREAT KISKADEE',
 'HOUSE FINCH']

def read_file_as_image(data) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return image

def create_json(images, segmented, predicted_classes, confidences, predicted, msgs):
    data = []
    for i in range(len(images)):
        if segmented:
            encode_image(images[i])
            item1 = {"class": predicted_classes[i], "confidence": str(confidences[i]), "predicted": predicted[i], "msg": msgs[i], "image": encode_image(images[i])}
            data.append(item1)
        else:
            item2 = {"class": "", "confidence": "", "predicted": "", "msg": "", "image": ''}
            data.append(item2)
    return data

def encode_image(image):
    _, im_arr = cv2.imencode('.jpg', image)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)

    return im_b64

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    predictions = []
    predicted_classes = []
    confidences = []
    predicted = []
    msgs = []
    segmented = False

    image = read_file_as_image(await file.read())

    images, full_image, labels = get_image_object(image)

    image_pre_resized = images.copy()

    for i in range(len(images)):

        if labels[i] == 'bird':
            images[i] = tf.image.resize(images[i], [224, 224])
            img_batch = np.expand_dims(images[i], 0)

            predictions = MODEL.predict(img_batch)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])

            if confidence < 0.8:
                predicted_class = ''
                confidence = 0.0
                msgs.append('Unrecognized animal predicted by the model')
            else:
                msgs.append('Recognized animal')

            predicted_classes.append(predicted_class)
            confidences.append(confidence)
            predicted.append(True)
            segmented = True

        elif labels[i] in CLASS_NAMES:
            predicted_class = ''
            predicted_classes.append(predicted_class)
            confidence = 0.0
            confidences.append(confidence)
            predicted.append(False)
            msgs.append('Unknown animal')
            segmented = True

        else:
            predicted_class = ''
            predicted_classes.append(predicted_class)
            confidence = 0.0
            confidences.append(confidence)
            predicted.append(False)
            msgs.append('No animal detected')

    data = create_json(image_pre_resized, segmented, predicted_classes, confidences, predicted, msgs)

    im_b64 = encode_image(full_image)

    return {
        'prediction' : data,
        "date": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        'segmented' : segmented,
        'full_image' : im_b64,
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8080)