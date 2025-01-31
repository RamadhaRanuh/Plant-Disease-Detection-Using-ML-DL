from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import json
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import requests

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
image_size = (IMAGE_HEIGHT, IMAGE_WIDTH)

app = FastAPI()

# "http://localhost:8600/v1/models/plantdisease_model/versions/4:predict"
# "http://localhost:8500/v1/models/plantdisease_model/labels/production:predict"

endpoint = "http://localhost:8500/v1/models/plantdisease_model:predict"

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

    json_data={
        "instances": img_batch.tolist() # float64
    }

    response = requests.post(url=endpoint, json=json_data)
    prediction = response.json()["predictions"][0]
    # predicted_class = np.argmax(prediction, axis=0)
    # np.argmax(prediction, axis=0) returns a numpy.int64 object, which is not JSON serializable.

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    # print(type(predicted_class))
    # print(type(int(predicted_class)))
    confidence = np.max(prediction)
    # print(type(float(confidence)))
    return {
        "class": predicted_class,
        "confidence": float(confidence) # float() float64
    }
"""
The numpy.int64 object cannot be directly serialized into JSON format by FastAPI.
By default, FastAPI tries to convert all Python objects into a format that can be sent over HTTP (usually JSON).
However, objectS like numpy.int64 cannot be directly converted into JSON
because they are not standard Python data types like int or float.
"""


if __name__ == "__main__":
    uvicorn.run(app, host="localhost",port=8080)