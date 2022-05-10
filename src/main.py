from controller.prediction_controller import predict_img
from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    return await predict_img(file=file)


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8080)
