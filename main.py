from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = load_model("model/mnist_cnn.h5")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    image_data = data["image"].split(",")[1]
    image_bytes = base64.b64decode(image_data)

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = 255 - img
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "digit": digit,
        "confidence": round(confidence * 100, 2)
    }
