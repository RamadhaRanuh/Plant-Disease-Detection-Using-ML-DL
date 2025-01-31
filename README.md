# Plant-Disease-Detection-ML_DL

#### Dataset Link: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

- ### Run with GPU:
  -> [Run Tensorflow on GPU.txt](https://github.com/Michs224/Plant-Disease-Detection-ML_DL/blob/main/requirements.txt)

## Prerequisites

### Clone the repository:
   ```
   git clone [https://github.com/your_username/your_repo.git](https://github.com/Michs224/Plant-Disease-Detection-ML_DL.git)
   cd Plant-Disease-Detection-ML_DL
   ```

### 1. Install Python: Follow the [Python Setup Instructions](https://www.python.org/downloads/).
### 2. Install additional dependencies:
  - ### Using Conda

    - Create and activate a Conda environment with Python 3.10:
       ```bash
       conda create --name myenv python=3.10
       conda activate myenv
       ```
    - Install dependencies:
       ```bash
       pip install -r Training/requirements.txt
       pip install -r api/requirements.txt
       ```

  - ### Using venv (Virtual Environment)

    - Create a virtual environment with Python 3.10:
       ```bash
       py -3.10 -m venv myenv
       ```
    - Activate the virtual environment:
      
      On Windows:
         ```bash
         myenv\Scripts\activate
         ```
      On macOS/Linux:
         ```bash
         source myenv/bin/activate
         ```
    - Install dependencies:
       ```bash
       pip install -r Training/requirements.txt
       pip install -r api/requirements.txt
       ```

### 3. **Install TensorFlow Serving**: Follow the [TensorFlow Serving Setup Instructions](https://www.tensorflow.org/tfx/guide/serving) or [Tensorflow Serving with Docker](https://www.tensorflow.org/tfx/serving/docker).

---

## Running the API

### Using FastAPI

1. **Navigate to the API folder**:
    ```bash
    cd api
    ```
2. **Run FastAPI Server**:
    ```bash
    uvicorn main:app --reload --host 0.0.0.0
    ```
3. Your API will now be running at `http://0.0.0.0:8080`.

### Using FastAPI and TF Serving

1. **Navigate to the API folder**:
    ```bash
    cd api
    ```
2. Copy `models.config.example` to `models.config` and update the paths.
3. **Run TensorFlow Serving**:
    ```bash
    docker run -it --name PlantDiseaseTFServing --rm -p 8500:8501 -v "Y:\Michh\Python\Projects\Plant Disease Detection":/Plant-Disease tensorflow/serving --rest_api_port=8501 --model_config_file=/Plant-Disease/models.config --allow_version_labels_for_unavailable_models
    ```
4. **Run FastAPI Server with TensorFlow Serving**:
    ```bash
    uvicorn main-tf-serving:app --reload --host 0.0.0.0
    ```
5. Your API will now be running at `http://0.0.0.0:8080`.

---

## Running the Apps

### Gradio App

1. **Navigate to the Gradio app folder and and install Gradio**:
    ```bash
    cd Gradio-app
    pip install -r requirements.txt
    ```
2. **Run the Gradio app**:
    ```bash
    python app.py
    ```

---

## Deploying the TensorFlow Model to GCP

### Steps for GCP Deployment:

1. **Create a GCP account**: If you don’t have one already, [sign up for Google Cloud Platform](https://cloud.google.com/).
2. **Create a new project**: 
    - Go to the [Google Cloud Console](https://console.cloud.google.com/).
    - Create a new project (make a note of the **project ID**).
3. **Create a [GCP bucket](https://console.cloud.google.com/storage/browser)**: 
4. **Upload the TensorFlow model to the GCP bucket**:
    - Upload the `.h5` model file to the bucket under the path `models/model_4.h5`.
5. **Install Google Cloud SDK**: Follow the [Google Cloud SDK Installation Guide](https://cloud.google.com/sdk/docs/install) to install the GCP SDK on your local machine.
6. **Authenticate with Google Cloud SDK**:
    ```bash
    gcloud auth login
    ```
7. **Run the deployment script**:
    - Clone the **GCP deployment script** from this repository:
    ```bash
    cd gcp
    ```
    - Deploy the function using the following command:
    ```bash
    gcloud functions deploy predict_plantdisease --runtime python310 --trigger-http --memory 2048 --timeout 540s --entry-point predict --project PROJECT_ID --allow-unauthenticated
    ```
    - Replace `PROJECT_ID` with your actual GCP project ID.

8. **Your model is now deployed on GCP**.
9. **Test the GCF using Postman**:
    - After deployment, you will receive a [Trigger URL](https://cloud.google.com/functions/docs/calling/http) for your function. Use **Postman** to send a
    - POST request to this URL for testing your model's inference.

---

## HuggingFace Space: https://huggingface.co/spaces/Mich24/Plant-Disease-Detection
