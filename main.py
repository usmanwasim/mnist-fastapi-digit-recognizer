from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import cv2
import base64
import os
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Auto-train if model missing
if not os.path.exists("model/mnist_lr.joblib"):
    print("Model not found. Training...")
    os.system("python train.py")

model = joblib.load("model/mnist_lr.joblib")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    image_data = data["image"].split(",")[1]
    image_bytes = base64.b64decode(image_data)

    # Preprocess image
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img.flatten() / 255.0

    digit = int(model.predict([img])[0])

    return {"digit": digit}
