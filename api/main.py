from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import json
from tensorflow.keras.applications.resnet_v2 import preprocess_input

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
image_size = (IMAGE_HEIGHT, IMAGE_WIDTH)

app = FastAPI()

MODEL = tf.keras.models.load_model("../Saved_models/4")

with open("../Saved_models/class_names.json", "r") as file:
    CLASS_NAMES = json.load(file)

@app.get("/hello")
async def ping():
    return "Hello, I am alive"

def preprocess_image(file) -> np.ndarray:
    image = Image.open(BytesIO(file))
    image = image.convert("RGB").resize(image_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = preprocess_input(image_array)
    return image_array


@app.post("/predict")
async def prediction(
    file: UploadFile
):
    bytes = await file.read()

    # print(bytes)
    # print(BytesIO)
    image = preprocess_image(bytes)
    img_batch = np.expand_dims(image, axis=0)
    # print(img_batch.dtype, img_batch[0][0][0][0]) # float32
    # print(type(img_batch.tolist()[0][0][0][0]), img_batch.tolist()[0][0][0][0]) # float64

    prediction = MODEL.predict(img_batch)
    pred_class = CLASS_NAMES[np.argmax(prediction[0], axis=0)]
    confidence = np.max(prediction[0])
    # confidence = np.float64(confidence)
    # print(type(float(confidence)))
    return {"class": pred_class, "confidence": float(confidence)}
    

if __name__ == "__main__":
    uvicorn.run(app, host="localhost",port=8080)