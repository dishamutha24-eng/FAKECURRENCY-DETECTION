
import copy
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "currency_resnet18.pth"

BATCH_SIZE = 4
EPOCHS = 12
SEED = 42
IMAGE_SIZE = 224

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")

print("Training device:", device)


# ==================================================
# 2. CHECK DATASET
# ==================================================

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR}"
    )

base_dataset = datasets.ImageFolder(DATA_DIR)

expected_classes = {"counterfeit", "genuine"}

if set(base_dataset.classes) != expected_classes:
    raise ValueError(
        "Inside dataset/, create exactly these folders:\n"
        "dataset/counterfeit/\n"
        "dataset/genuine/"
    )

if len(base_dataset.classes) != 2:
    raise ValueError("Exactly two classes are required.")

print("\nClass mapping:", base_dataset.class_to_idx)
print("Total images:", len(base_dataset))

for class_name, class_index in base_dataset.class_to_idx.items():
    count = sum(
        label == class_index
        for _, label in base_dataset.samples
    )
    print(f"{class_name}: {count} images")

targets = [label for _, label in base_dataset.samples]
class_counts = Counter(targets)

if min(class_counts.values()) < 4:
    raise ValueError(
        "Each class needs at least four images "
        "for stratified train/validation splitting."
    )


# ==================================================
# 3. IMAGE TRANSFORMS
# ==================================================

# Keep augmentation mild to preserve currency-note details.

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(3),
    transforms.ColorJitter(
        brightness=0.08,
        contrast=0.08,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ==================================================
# 4. STRATIFIED TRAIN / VALIDATION SPLIT
# ==================================================

indices = list(range(len(base_dataset)))

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.25,
    random_state=SEED,
    stratify=targets,
)

print(f"\nTraining images: {len(train_indices)}")
print(f"Validation images: {len(val_indices)}")


# ==================================================
# 5. DATASET
# ==================================================

class CurrencyDataset(Dataset):

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        actual_index = self.indices[index]

        image_path, label = self.dataset.samples[actual_index]

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        image = self.transform(image)

        return image, label


train_dataset = CurrencyDataset(
    base_dataset,
    train_indices,
    train_transform,
)

val_dataset = CurrencyDataset(
    base_dataset,
    val_indices,
    eval_transform,
)


# ==================================================
# 6. BALANCED SAMPLING
# ==================================================

train_labels = [
    base_dataset.samples[index][1]
    for index in train_indices
]

train_counts = Counter(train_labels)

sample_weights = [
    1.0 / train_counts[label]
    for label in train_labels
]

sampler_generator = torch.Generator().manual_seed(SEED)

train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
    generator=sampler_generator,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ==================================================
# 7. LOAD PRETRAINED RESNET18
# ==================================================

print("\nLoading pretrained ResNet18...")

try:
    model = models.resnet18(
        weights=ResNet18_Weights.DEFAULT
    )
except Exception as error:
    raise RuntimeError(
        "Could not load pretrained ResNet18 weights. "
        "Check your internet connection or whether the "
        "pretrained weights are already cached."
    ) from error


# Freeze the pretrained feature extractor.
for parameter in model.parameters():
    parameter.requires_grad = False


# Replace the final classifier with a two-class classifier.
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.fc.in_features, 2),
)

# Only the final classifier will be trained.
for parameter in model.fc.parameters():
    parameter.requires_grad = True

model = model.to(device)


# ==================================================
# 8. LOSS AND OPTIMIZER
# ==================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.0005,
    weight_decay=0.001,
)


# ==================================================
# 9. TRAINING
# ==================================================

best_val_loss = float("inf")
best_state = copy.deepcopy(model.state_dict())
best_epoch = 0

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / max(total, 1)
    train_accuracy = 100.0 * correct / max(total, 1)

    # ------------------------------
    # VALIDATION
    # ------------------------------

    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = val_loss_total / max(val_total, 1)
    val_accuracy = 100.0 * val_correct / max(val_total, 1)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.1f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.1f}%"
    )

    # Keep the model with the lowest validation loss.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        best_state = copy.deepcopy(model.state_dict())


# ==================================================
# 10. RESTORE AND SAVE BEST MODEL
# ==================================================

model.load_state_dict(best_state)
model.eval()

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_to_idx": base_dataset.class_to_idx,
        "image_size": IMAGE_SIZE,
    },
    MODEL_PATH,
)

print("\nTraining complete.")
print("Best epoch:", best_epoch)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Model saved to: {MODEL_PATH}")


# ==================================================
# 11. VALIDATION REPORT
# ==================================================

true_labels = []
predicted_labels = []

model.eval()

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().tolist()

        true_labels.extend(labels.tolist())
        predicted_labels.extend(predictions)

idx_to_class = {
    index: name
    for name, index in base_dataset.class_to_idx.items()
}

ordered_names = [
    idx_to_class[index]
    for index in range(len(idx_to_class))
]

print("\nValidation classification report:")

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(ordered_names))),
        target_names=ordered_names,
        zero_division=0,
    )
)

print("\nConfusion matrix:")

print(
    confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(ordered_names))),
    )
)

print(
    "\nValidation metrics use a small dataset and "
    "may not reflect performance on new images."
)
