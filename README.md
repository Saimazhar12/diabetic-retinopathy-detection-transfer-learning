# 👁️ Diabetic Retinopathy Detection using Fine-Tuned Deep Learning Model

This project implements a deep learning-based system to detect Diabetic Retinopathy (DR) from retinal images using transfer learning and fine-tuned CNN models. The model is trained on retinal fundus images and is capable of accurately classifying whether an eye is healthy or affected by DR.

## 📌 Project Overview
The goal of this project is to build an automated classifier that identifies Diabetic Retinopathy from retinal images. This helps in early diagnosis and prevention of vision loss. Pre-trained CNN models are used to automatically learn important visual features.

## 📂 Dataset
- Dataset Type: Retinal Fundus Images  
- Classes: DR (Diseased), No DR (Healthy) 
- Images are preprocessed and resized for training  

## ⚙️ Features
- Transfer Learning (InceptionV3)  
- Fine-tuning for improved performance  
- Data preprocessing and augmentation  
- Regularization using Dropout  
- EarlyStopping and ModelCheckpoint  

## 🧠 Model & Training
- Optimizer: Adam  
- Loss Function: Binary/Categorical Crossentropy  
- Batch Size: 32  
- Epochs: 20  

## 📊 Results
The model achieves strong performance on validation data and generalizes well to unseen retinal images.

## 🛠️ Technologies Used
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Scikit-learn  

## 🚀 How to Run
1. Clone the repository  
2. Install dependencies
3. Open Google Colab
4. Run the all cells step by step  

## 📌 Conclusion
This project highlights the effectiveness of deep learning in medical image analysis and can be further improved using advanced architectures and larger datasets.

## 🔮 Future Improvements
- Use larger and more diverse datasets for better generalization  
- Implement multi-class classification for DR severity levels  
- Deploy as a web application using Flask or Fast API
- Optimize model for real-time predictions  
 
