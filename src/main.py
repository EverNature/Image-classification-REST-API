from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from datetime import datetime
from opencv import get_image_object

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
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())

    image, label = get_image_object(image)

    if label == 'bird':
        image = tf.image.resize(image, (224, 224))
        img_batch = np.expand_dims(image, 0)
        
        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        predicted = True
        msg = f"{predicted_class} {confidence}"

    elif label in LABEL_NAMES:
        predicted_class = ''
        confidence = 0.0
        predicted = False
        msg = 'Unknown animal'

    else:
        predicted_class = ''
        confidence = 0.0
        predicted = False
        msg = 'No animal detected'
    
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'date': datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        'predicted': predicted,
        'msg': msg
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8080)