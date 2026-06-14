from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import numpy as np
import os

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# load model
model = load_model("DR_InceptionV3.h5")
print('@@ Model loaded')


def pred_diabetic_retinopathy(dr_image):
    test_image = load_img(dr_image, target_size=(299, 299))
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255.0
    test_image = np.expand_dims(test_image, axis=0)

    result = model.predict(test_image)[0][0]
    print('@@ Raw result = ', result)

    if result < 0.5:
        return "Diabetic Retinopathy Detected (DR)"
    else:
        return "No Diabetic Retinopathy (No_DR)"


# Create app
app = FastAPI()

# Static + Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Home route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


# Predict route
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, image: UploadFile = File(...)):

    filename = image.filename
    print("@@ Input posted = ", filename)

    upload_folder = 'static/user_uploaded'
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await image.read())

    print("@@ Predicting class......")
    pred = pred_diabetic_retinopathy(file_path)

    return templates.TemplateResponse(request, "predict.html", {
        "pred_output": pred,
        "user_image": file_path
    })
