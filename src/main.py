from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from datetime import datetime


app = FastAPI()

MODEL = tf.keras.models.load_model("modelo/vgg16_weights_full_model")

CLASS_NAMES = ['ABBOTTS BABBLER',
 'ABBOTTS BOOBY',
 'AFRICAN FIREFINCH',
 'AFRICAN OYSTER CATCHER',
 'BEARDED BARBET',
 'CASPIAN TERN',
 'CEDAR WAXWING',
 'GREAT KISKADEE',
 'HOUSE FINCH']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image = tf.image.resize(image, (224, 224))
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'date': datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)