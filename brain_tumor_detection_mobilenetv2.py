Code
# Import

import os
import zipfile
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from sklearn.metrics import classification_report, confusion_matrix
import kagglehub
path1 = kagglehub.dataset_download("preetviradiya/brian-tumor-dataset")
print("Dataset 1 Path:", path1)
path2 = kagglehub.dataset_download("sartajbhuvaji/brain-tumor-classification-mri")
print("Dataset 2 Path:", path2)
path3 = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")
print("Dataset 3 Path:", path3)
# Data Augmentation

dataset_root = "/kaggle/input/brian-tumor-dataset/Brain Tumor Data Set/Brain Tumor Data Set"
img_size = 128  # smaller images for CPU
batch_size = 16

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_set = datagen.flow_from_directory(
    dataset_root,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_set = datagen.flow_from_directory(
    dataset_root,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze base

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)  # Binary classification

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
history = model.fit(
    train_set,
    validation_data=val_set,
    epochs=5  # quick CPU-friendly run
)
from google.colab import drive
drive.mount('/content/drive')
# ---------------------------
# Evaluate Model
# ---------------------------
loss, accuracy = model.evaluate(val_set)
print(f"\n✔ Final Accuracy: {accuracy*100:.2f}%")
print(f"❌ Final Loss: {loss:.4f}")
# ---------------------------
#  Predictions & Metrics
# ---------------------------
val_set.reset()
predictions = model.predict(val_set)
y_pred = (predictions > 0.5).astype(int).reshape(-1)

print("\n🧾 Classification Report:")
print(classification_report(val_set.classes, y_pred, target_names=list(train_set.class_indices.keys())))

cm = confusion_matrix(val_set.classes, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(train_set.class_indices.keys()),
            yticklabels=list(train_set.class_indices.keys()))
plt.title("Confusion Matrix")
plt.show()

# ---------------------------
# Accuracy & Loss Plots
# ---------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Graph")
plt.show()
# =============================
# MULTI-DATASET PREDICTION
# =============================
base_predict_dir = "/content/sample_data/multi_predict"  # parent folder containing dataset1, dataset2...

confidence_threshold = 0.5
datagen = ImageDataGenerator(rescale=1./255)

idx_to_class = {v: k for k, v in train_set.class_indices.items()}

print("\n===== MULTI DATASET PREDICTIONS =====\n")

for dataset_name in os.listdir(base_predict_dir):
    if dataset_name.startswith("."):
        continue
    dataset_path = os.path.join(base_predict_dir, dataset_name)

    if os.path.isdir(dataset_path):
        print(f"\n Processing {dataset_name}...\n")

        generator = datagen.flow_from_directory(
            directory=base_predict_dir,
            classes=[dataset_name],   # Only load this dataset folder
            target_size=(img_size, img_size),
            batch_size=1,
            class_mode=None,
            shuffle=False
        )

        predictions = model.predict(generator)
        filenames = generator.filenames

        for i, pred in enumerate(predictions):
            filename = os.path.basename(filenames[i])
            pred_class = (pred > confidence_threshold).astype(int)[0]

            if pred_class == 0:
                label = "Brain Tumor"
                confidence = 1 - pred[0]
            else:
                label = "Healthy"
                confidence = pred[0]

            print(f"Image: {filename:<20} | Predicted: {label:<12} | Confidence: {confidence:.4f}")


