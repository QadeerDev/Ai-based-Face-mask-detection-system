"""
Real-Time Face Mask Detection - Training Script
Author: Muhammad Qadeer | github.com/QadeerDev
Day 3: LinkedIn AI Portfolio Series

Architecture: MobileNetV2 (Transfer Learning)
Dataset: Face Mask Detection ~4000 images (with_mask / without_mask)
Use Case: Medical AI - Hospital / Clinical Compliance Monitoring
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Deep Learning
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_PATH   = "./data"          # expects data/with_mask & data/without_mask
MODEL_SAVE     = "./models/mask_detector.keras"
PLOT_SAVE      = "./results/training_curves.png"
CM_SAVE        = "./results/confusion_matrix.png"

INIT_LR        = 1e-4
EPOCHS         = 20
BATCH_SIZE     = 32
IMG_SIZE       = (224, 224)
CLASSES        = ["with_mask", "without_mask"]
SEED           = 42

# ─────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────
def build_data_generators(dataset_path: str):
    """Build train/val generators with aggressive augmentation."""

    # Medical-grade augmentation: simulate different lighting, angles, partial occlusion
    train_aug = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],   # simulate hospital lighting variations
        fill_mode="nearest",
        validation_split=0.20
    )

    val_aug = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.20
    )

    train_gen = train_aug.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        seed=SEED,
        shuffle=True
    )

    val_gen = val_aug.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        seed=SEED,
        shuffle=False
    )

    print(f"\n✅ Classes found  : {train_gen.class_indices}")
    print(f"   Train samples  : {train_gen.samples}")
    print(f"   Val   samples  : {val_gen.samples}")
    return train_gen, val_gen


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────
def build_model():
    """
    MobileNetV2 + custom classification head.
    - Freeze all base layers during initial training
    - Fine-tune last 30 layers in second pass (optional)
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(224, 224, 3))
    )
    # Freeze base
    base_model.trainable = False

    # Custom head optimized for binary classification
    head = base_model.output
    head = AveragePooling2D(pool_size=(7, 7))(head)
    head = Flatten(name="flatten")(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(1, activation="sigmoid")(head)   # binary output

    model = Model(inputs=base_model.input, outputs=head)

    model.compile(
        optimizer=Adam(learning_rate=INIT_LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    total    = len(model.layers)
    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"\n🧠 Model built  | Total layers: {total} | Trainable: {trainable}")
    return model


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train_model(model, train_gen, val_gen):
    """Train with smart callbacks."""

    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print(f"\n🚀 Training started | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    return history


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
def plot_training_curves(history):
    """Publication-quality training curves."""
    os.makedirs("./results", exist_ok=True)

    fig = plt.figure(figsize=(16, 5), facecolor="#0D1117")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    metrics = [
        ("accuracy",    "Model Accuracy",     "#00D4FF", "#FF6B35"),
        ("loss",        "Training Loss",       "#00FF94", "#FF4757"),
        ("auc",         "ROC AUC Score",       "#C471ED", "#F7971E"),
    ]

    for i, (metric, title, c1, c2) in enumerate(metrics):
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor("#161B22")
        ax.spines[["top","right","left","bottom"]].set_color("#30363D")

        train_key = metric
        val_key   = f"val_{metric}"

        if train_key in history.history:
            epochs = range(1, len(history.history[train_key]) + 1)
            ax.plot(epochs, history.history[train_key], color=c1,
                    linewidth=2.5, label=f"Train", zorder=3)
            ax.fill_between(epochs, history.history[train_key],
                            alpha=0.12, color=c1)

        if val_key in history.history:
            ax.plot(epochs, history.history[val_key], color=c2,
                    linewidth=2.5, linestyle="--", label=f"Validation", zorder=3)

        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch", color="#8B949E", fontsize=10)
        ax.tick_params(colors="#8B949E", labelsize=9)
        ax.legend(facecolor="#0D1117", edgecolor="#30363D",
                  labelcolor="white", fontsize=9)
        ax.grid(True, color="#21262D", linewidth=0.6, alpha=0.7)

    fig.suptitle("Face Mask Detector — Training Results",
                 color="white", fontsize=16, fontweight="bold", y=1.02)

    plt.savefig(PLOT_SAVE, dpi=150, bbox_inches="tight",
                facecolor="#0D1117", edgecolor="none")
    print(f"\n📊 Training curves saved → {PLOT_SAVE}")
    plt.close()


def plot_confusion_matrix(model, val_gen):
    """Confusion matrix with medical-grade layout."""
    val_gen.reset()
    preds = model.predict(val_gen, verbose=0)
    y_pred = (preds > 0.5).astype(int).flatten()
    y_true = val_gen.classes

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")

    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        linewidths=1, linecolor="#21262D",
        annot_kws={"size": 18, "weight": "bold", "color": "white"},
        ax=ax
    )

    ax.set_title("Confusion Matrix — Face Mask Detector",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", color="#8B949E", fontsize=11)
    ax.set_ylabel("True Label", color="#8B949E", fontsize=11)
    ax.tick_params(colors="white", labelsize=10)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print("\n📋 Classification Report:\n")
    print(report)

    plt.tight_layout()
    plt.savefig(CM_SAVE, dpi=150, bbox_inches="tight",
                facecolor="#0D1117", edgecolor="none")
    print(f"📊 Confusion matrix saved → {CM_SAVE}")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  🏥 Real-Time Face Mask Detector — Training")
    print("  Architecture : MobileNetV2 (Transfer Learning)")
    print("  Use Case     : Medical AI / Clinical Compliance")
    print("=" * 55)

    os.makedirs("./models",  exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    if not Path(DATASET_PATH).exists():
        print(f"\n❌ Dataset not found at '{DATASET_PATH}'")
        print("   Download from: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset")
        print("   Expected structure:")
        print("     data/")
        print("     ├── with_mask/    (~2000 images)")
        print("     └── without_mask/ (~2000 images)")
        return

    train_gen, val_gen = build_data_generators(DATASET_PATH)
    model = build_model()
    history = train_model(model, train_gen, val_gen)

    # Evaluate
    print("\n🔍 Evaluating on validation set...")
    loss, acc, auc = model.evaluate(val_gen, verbose=0)
    print(f"\n✅ Final Results:")
    print(f"   Val Accuracy : {acc*100:.2f}%")
    print(f"   Val AUC      : {auc:.4f}")
    print(f"   Val Loss     : {loss:.4f}")

    plot_training_curves(history)
    plot_confusion_matrix(model, val_gen)

    print(f"\n💾 Model saved → {MODEL_SAVE}")
    print("\n🎉 Training complete! Run `python src/detect_realtime.py` for live demo.")


if __name__ == "__main__":
    main()
