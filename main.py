import os
import shutil

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from ml_service import train_model, predict_image

app = FastAPI(title="Skin Cancer Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Skin Cancer Detection Backend Running"
    }


@app.get("/models")
def get_models():
    return {
        "available_models": ["resnet18", "resnet50", "vgg16","DENSENET121","MOBILENETV2","EFFICIENTNET B0","VISION TRANSFORMER (ViT)","SWIN TRANSFORMER","ConvNeXt Tiny"]
    }


@app.post("/train")
def train(
    model_name: str = Form(...),
    epochs: int = Form(...),
    train_split: float = Form(...),
    lr: float = Form(...)
):
    result = train_model(
        model_name=model_name,
        epochs=epochs,
        train_split=train_split,
        lr=lr
    )

    return result


@app.post("/predict")
def predict(
    model_id: str = Form(...),
    file: UploadFile = File(...)
):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    image_path = os.path.join(upload_dir, file.filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_image(model_id, image_path)

    return result
