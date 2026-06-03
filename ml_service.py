import os
import uuid
import time
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Change this path to your dataset folder
DATASET_PATH = r"C:\Users\SARAVANA\anaconda3\projects\skin-cancer-app\dataset\Split_smol"

SAVE_DIR = "models_saved"
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES = []


def log(message):
    print(f"[BACKEND] {message}", flush=True)


def load_data():
    log("Loading dataset...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_path = os.path.join(DATASET_PATH, "train")
    val_path = os.path.join(DATASET_PATH, "val")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train folder not found: {train_path}")

    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation folder not found: {val_path}")

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    val_dataset = datasets.ImageFolder(val_path, transform=transform)

    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            f"Train and val class folders do not match. "
            f"Train classes: {train_dataset.classes}, "
            f"Val classes: {val_dataset.classes}"
        )

    global CLASS_NAMES
    CLASS_NAMES = train_dataset.classes

    log(f"Train path: {train_path}")
    log(f"Val path: {val_path}")
    log(f"Classes found: {CLASS_NAMES}")
    log(f"Train images: {len(train_dataset)}")
    log(f"Validation images: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    log("Data loaders created successfully.")

    return train_loader, val_loader, len(CLASS_NAMES)


def build_model(model_name: str, num_classes: int):
    log(f"Building model: {model_name}")
    log(f"Number of classes: {num_classes}")

    # ResNet18
    if model_name == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )

    # ResNet50
    elif model_name == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )
        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )

    # DenseNet121
    elif model_name == "densenet121":
        model = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )
        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes
        )

    # MobileNetV2
    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

    # EfficientNet-B0
    elif model_name == "efficientnetb0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

    # AlexNet
    elif model_name == "alexnet":
        model = models.alexnet(
            weights=models.AlexNet_Weights.DEFAULT
        )
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features,
            num_classes
        )

    # Vision Transformer (ViT)
    elif model_name == "vit":
        model = models.vit_b_16(
            weights=models.ViT_B_16_Weights.DEFAULT
        )
        model.heads.head = nn.Linear(
            model.heads.head.in_features,
            num_classes
        )

    # Swin Transformer
    elif model_name == "swin":
        model = models.swin_t(
            weights=models.Swin_T_Weights.DEFAULT
        )
        model.head = nn.Linear(
            model.head.in_features,
            num_classes
        )

    # ConvNeXt Tiny
    elif model_name == "convnext":
        model = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        )
        model.classifier[2] = nn.Linear(
            model.classifier[2].in_features,
            num_classes
        )

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    model = model.to(DEVICE)

    log(f"Model moved to device: {DEVICE}")
    log("Model created successfully.")

    return model


def evaluate_model(model, val_loader):
    log("Evaluating model on validation data...")

    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_accuracy = 100 * correct / total if total > 0 else 0

    log(f"Validation loss: {val_loss:.4f}")
    log(f"Validation accuracy: {val_accuracy:.2f}%")

    return val_loss, val_accuracy


def train_model(model_name: str, epochs: int, train_split: float, lr: float):
    start_time = time.time()

    log("=" * 60)
    log("TRAINING STARTED")
    log(f"Selected model: {model_name}")
    log(f"Epochs: {epochs}")
    log("Train split ignored because dataset already has train/val folders")
    log(f"Learning rate: {lr}")
    log(f"Using device: {DEVICE}")
    log("=" * 60)

    train_loader, val_loader, num_classes = load_data()

    model = build_model(model_name, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    log("Loss function: CrossEntropyLoss")
    log("Optimizer: Adam")
    log("Starting training loop...")

    history = []

    total_batches = len(train_loader)

    for epoch in range(epochs):
        epoch_start_time = time.time()

        log("-" * 60)
        log(f"Epoch {epoch + 1}/{epochs} started")

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_index % 5 == 0 or batch_index == total_batches:
                running_accuracy = 100 * correct / total

                log(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Batch {batch_index}/{total_batches} | "
                    f"Current loss: {loss.item():.4f} | "
                    f"Running accuracy: {running_accuracy:.2f}%"
                )

        train_loss = running_loss / total_batches
        train_accuracy = 100 * correct / total

        val_loss, val_accuracy = evaluate_model(model, val_loader)

        epoch_time = time.time() - epoch_start_time

        log(f"Epoch {epoch + 1}/{epochs} completed")
        log(f"Training loss: {train_loss:.4f}")
        log(f"Training accuracy: {train_accuracy:.2f}%")
        log(f"Validation loss: {val_loss:.4f}")
        log(f"Validation accuracy: {val_accuracy:.2f}%")
        log(f"Epoch time: {epoch_time:.2f} seconds")

        history.append({
            "epoch": epoch + 1,
            "loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

    model_id = str(uuid.uuid4())
    model_path = os.path.join(SAVE_DIR, f"{model_id}.pth")

    log("Saving trained model...")

    torch.save({
        "model_name": model_name,
        "num_classes": num_classes,
        "class_names": CLASS_NAMES,
        "state_dict": model.state_dict()
    }, model_path)

    total_time = time.time() - start_time

    log("=" * 60)
    log("TRAINING COMPLETED")
    log(f"Model ID: {model_id}")
    log(f"Model saved at: {model_path}")
    log(f"Total training time: {total_time:.2f} seconds")
    log("=" * 60)

    return {
        "message": "Training completed",
        "model_id": model_id,
        "model_path": model_path,
        "history": history
    }


def predict_image(model_id: str, image_path: str):
    log("=" * 60)
    log("PREDICTION STARTED")
    log(f"Model ID: {model_id}")
    log(f"Input image: {image_path}")

    checkpoint_path = os.path.join(SAVE_DIR, f"{model_id}.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")

    log(f"Loading model checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model = build_model(
        checkpoint["model_name"],
        checkpoint["num_classes"]
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    log("Model loaded successfully.")
    log(f"Model classes: {checkpoint['class_names']}")
    log("Processing input image...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    log("Running prediction...")

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = checkpoint["class_names"][predicted.item()]
    confidence_value = float(confidence.item())

    log(f"Prediction: {predicted_class}")
    log(f"Confidence: {confidence_value * 100:.2f}%")
    log("PREDICTION COMPLETED")
    log("=" * 60)

    return {
        "prediction": predicted_class,
        "confidence": confidence_value
    }
