import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --------------------
# CONFIG
# --------------------
DATA_DIR = r"C:\Users\rudra\Desktop\My_Pain_Study\all_faces_dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10   # Train only top layers
EPOCHS_PHASE2 = 10   # Fine-tune deeper layers
MODEL_PATH = "pain_face_transfer_best.h5"

# --------------------
# DATA GENERATORS
# --------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# --------------------
# MODEL
# --------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # freeze first for phase 1

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# --------------------
# CALLBACKS
# --------------------
checkpoint = ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# --------------------
# PHASE 1: Train top layers only
# --------------------
print("\n🔹 Phase 1: Training top layers...")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE1,
    callbacks=[checkpoint, earlystop]
)

# --------------------
# PHASE 2: Fine-tune deeper layers
# --------------------
print("\n🔹 Phase 2: Fine-tuning deeper layers...")

base_model.trainable = True
# Freeze first N layers → fine-tune only deeper ones
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # lower LR for fine-tuning
              loss="binary_crossentropy", metrics=["accuracy"])

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE2,
    callbacks=[checkpoint, earlystop]
)

print(f"\n✅ Training complete. Best model saved as {MODEL_PATH}")
