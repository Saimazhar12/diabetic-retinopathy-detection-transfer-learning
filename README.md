# 👁️ Diabetic Retinopathy Detection using Deep Learning (InceptionV3)

This project implements a deep learning-based system to detect **Diabetic Retinopathy (DR)** from retinal fundus images using a fine-tuned **InceptionV3** model. The model is trained on retinal images and achieves high performance in classifying whether Diabetic Retinopathy is present or not.

# 🌐 Live Demo

👉 [Open Hugging Face App](https://huggingface.co/spaces/Saim1598/diabetic-retinopathy-detector)

# 📌 Project Overview

The goal of this project is to build a binary image classification model that can accurately detect Diabetic Retinopathy from retinal fundus images. The project includes image preprocessing, data augmentation, transfer learning, fine-tuning, model evaluation, and deployment using FastAPI.

# 📂 Dataset

- Total Images: 5,676
- Training Images: 2,076
- Validation Images: 531
- Testing Images: 231
- Target Classes:
  - Diabetic Retinopathy (DR)
  - No Diabetic Retinopathy (No_DR)
- Image Size: 299 × 299

# ⚙️ Features

- Retinal fundus image classification
- Image preprocessing and normalization
- Data augmentation for improved generalization
- Transfer Learning using InceptionV3
- Fine-tuning of the last 50 layers
- Early Stopping and Model Checkpointing
- Model evaluation using Classification Report and Confusion Matrix
- FastAPI web application for real-time predictions
- Hugging Face deployment

# 🧠 Model & Training

- Model: InceptionV3 (ImageNet Pre-trained)
- Transfer Learning Applied
- Fine-Tuning: Last 50 layers unfrozen
- Optimizer: Adam
- Learning Rate: 1e-5
- Loss Function: Binary Crossentropy
- Batch Size: 32
- Epochs: 20
- Input Image Size: 299 × 299

# 📊 Results

The model achieves ~97% accuracy on unseen test data in distinguishing retinal images with and without Diabetic Retinopathy.

# 🌍 Web Application Deployment (FastAPI)

The trained InceptionV3 model was deployed as a full-stack web application using FastAPI and Hugging Face Spaces.

- Developed a FastAPI backend for real-time predictions
- Designed a responsive HTML/Bootstrap frontend
- Integrated the trained model using TensorFlow/Keras
- Created a `/predict` endpoint for image classification
- Implemented image upload functionality
- Displayed uploaded image and prediction result
- Deployed the complete application on Hugging Face Spaces

# 🧩 Project Structure

- `app.py` → FastAPI backend code
- `templates/index.html` → Home page UI
- `templates/predict.html` → Prediction result page
- `DR_InceptionV3.h5` → Trained Deep Learning model
- `static/images/` → Application images
- `static/user_uploaded/` → Uploaded images
- `requirements.txt` → Dependencies
- `notebook.ipynb` → Model training and evaluation

# 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- InceptionV3
- NumPy
- Matplotlib
- Scikit-learn
- FastAPI
- HTML
- Bootstrap
- Jinja2 Templates
- Hugging Face Spaces

# 🚀 How to Run

1. Clone the repository
2. Install dependencies using `pip install -r requirements.txt`
3. Run the FastAPI app using `uvicorn app:app --reload`
4. Open `http://127.0.0.1:8000`
5. Upload a retinal image and get predictions

# 📌 Conclusion

This project demonstrates the effectiveness of Transfer Learning with InceptionV3 for medical image classification. The model achieves high accuracy in detecting Diabetic Retinopathy and is deployed using FastAPI on Hugging Face Spaces, allowing users to perform real-time retinal image analysis through a web interface.
