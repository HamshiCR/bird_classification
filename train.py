from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import torchvision.transforms as transforms
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, RandomHorizontalFlip
import numpy as np
from PIL import Image

# 1. Load Dataset (folder must be structured as: dataset/class_name/images)
dataset = load_dataset("imagefolder", data_dir="segmentations")

# 2. Load Preprocessing Model
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

# 3. Define Image Transformations
normalize = Normalize(mean=processor.image_mean, std=processor.image_std)

train_transforms = Compose([
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
    ToTensor(),
    normalize,
])

val_transforms = Compose([
    transforms.Resize((224, 224)),
    ToTensor(),
    normalize,
])

def preprocess_train(examples):
    examples["pixel_values"] = [train_transforms(Image.open(img).convert("RGB")) for img in examples["image"]]
    return examples

def preprocess_val(examples):
    examples["pixel_values"] = [val_transforms(Image.open(img).convert("RGB")) for img in examples["image"]]
    return examples

# Apply transforms
train_ds = dataset["train"].with_transform(preprocess_train)
val_ds = dataset["test"].with_transform(preprocess_val)

# 4. Load Model for Training
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(dataset["train"].features["label"].names),
    ignore_mismatched_sizes=True
)

# 5. Training Settings
training_args = TrainingArguments(
    output_dir="./bird_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

# 7. Train Model
trainer.train()

# 8. Save Model
model.save_pretrained("./fine_tuned_bird_classifier")
processor.save_pretrained("./fine_tuned_bird_classifier")