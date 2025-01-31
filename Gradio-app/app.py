import gradio as gr
import requests
import io

def predict_plant_disease(image):
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        url = "https://us-central1-plant-disease-system.cloudfunctions.net/predict_plantdisease"
        response = requests.post(url, files={"file": img_byte_arr})

        if response.status_code == 200:
            result = response.json()
            plant_class = result["class"]
            confidence = result["confidence"]

            if confidence < 0.5:
                return image, "Error: Unable to classify the image. Please upload a clear image of the plant leaf."
            return image, f"Class: {plant_class}\nConfidence: {confidence}%"
        else:
            return image, "Error: Unable to get a valid response from the API."
    except Exception as e:
        return image, f"Error: {str(e)}"

# Sample
sample_images = [
    "sample/Apple_black_rot.jpg",
    "sample/Tomato_septoria_leaf_spot.jpg",
    "sample/Potato_healthy.jpg",
    "sample/Strawberry_leaf_scorch.jpg",
    "sample/Peach_bacterial_spot.jpg",
    "sample/Corn_(maize)__cercospora_leaf_spot gray_leaf_spot.jpg",
    "sample/Raspberry__healthy.jpg",
    "sample/Grape__esca_(black_measles).jpg"

]

PlantAvailable = ["Apple", "Tomato", "Potato", "Strawberry", "And more..."]

# UI Gradio
interface = gr.Interface(
    fn=predict_plant_disease,
    inputs=gr.Image(type="pil", label="Upload a clear image of the plant leaf"),
    outputs=[gr.Image(type="pil", label="Uploaded Plant Leaf Image"), gr.Textbox(label="Prediction")],
    live=True,
    title="Plant Disease Detection",
    description=f"""
        Upload an image of the plant leaf to predict the disease and see the confidence level.

        **Available Plants**:
        {', '.join(PlantAvailable)}

        The model will identify the disease affecting the plant.

        **Sample Images**:
    """,
    examples=sample_images
)

interface.launch()
