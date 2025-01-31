from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
import json

BUCKET_NAME = "plant_disease_tf_models"
with open("class_names.json", "r") as file:
    CLASS_NAMES = json.load(file)

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

model = None

def download_blob(bucket_name, source_blob_name, destination_file_name):
    # print(f"Downloading {source_blob_name} to {destination_file_name}")
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    # print("Download complete")

def predict(request):
    global model

    if model is None:
        download_blob(BUCKET_NAME,
                      "models/model_4.h5",
                      "/tmp/model_4.h5"
        )
        model = tf.keras.models.load_model("/tmp/model_4.h5")
    image = request.files["file"]
    image = Image.open(image).convert("RGB").resize(IMAGE_SIZE)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.resnet_v2.preprocess_input(image)
    img_batch = np.expand_dims(image, axis=0)

    prediction = model.predict(img_batch)
    pred_class = CLASS_NAMES[np.argmax(prediction[0], axis=0)]
    confidence = round(100 * np.max(prediction[0]), 3)

    return {"class": pred_class, "confidence": float(confidence)}
