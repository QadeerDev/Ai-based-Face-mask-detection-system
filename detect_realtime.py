"""
Real-Time Face Mask Detection — Live Inference
Author: Muhammad Qadeer | github.com/QadeerDev

Runs on webcam or video file.
Green box  = Mask detected ✅
Red box    = No mask ❌
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH    = "./models/mask_detector.keras"
FACE_PROTO    = "./models/deploy.prototxt"
FACE_WEIGHTS  = "./models/res10_300x300_ssd_iter_140000.caffemodel"

# OpenCV DNN face detector — download if missing
FACE_PROTO_URL   = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_WEIGHTS_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

CONF_THRESHOLD = 0.5   # face detection confidence
MASK_THRESHOLD = 0.5   # mask classification threshold

# Color palette
COLORS = {
    "mask":    (0, 204, 102),   # green
    "no_mask": (0, 51, 204),    # red  (BGR)
    "overlay": (13, 17, 23),    # dark bg
    "text":    (255, 255, 255),
    "accent":  (0, 212, 255),
}

LABELS = {0: "Mask ✓", 1: "No Mask ✗"}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def download_face_model():
    """Auto-download OpenCV SSD face detector if not present."""
    import urllib.request
    os.makedirs("./models", exist_ok=True)

    if not Path(FACE_PROTO).exists():
        print("📥 Downloading face detector prototxt...")
        urllib.request.urlretrieve(FACE_PROTO_URL, FACE_PROTO)

    if not Path(FACE_WEIGHTS).exists():
        print("📥 Downloading face detector weights (~2MB)...")
        urllib.request.urlretrieve(FACE_WEIGHTS_URL, FACE_WEIGHTS)
    print("✅ Face detector ready.")


def load_models():
    """Load both face detector and mask classifier."""
    print("🔄 Loading models...")
    face_net = cv2.dnn.readNet(FACE_PROTO, FACE_WEIGHTS)
    mask_net = load_model(MODEL_PATH)
    print("✅ Models loaded.\n")
    return face_net, mask_net


def detect_and_classify(frame, face_net, mask_net):
    """
    Detect faces in frame → classify each as mask/no-mask.
    Returns list of (startX, startY, endX, endY, label, confidence, color)
    """
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    results = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < CONF_THRESHOLD:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        # Extract face ROI
        face = frame[startY:endY, startX:endX]
        if face.size == 0:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        pred = mask_net.predict(face, verbose=0)[0][0]

        # pred close to 0 = with_mask, close to 1 = without_mask
        if pred < MASK_THRESHOLD:
            label = "Mask ✓"
            color = COLORS["mask"]
            conf  = 1 - pred
        else:
            label = "No Mask ✗"
            color = COLORS["no_mask"]
            conf  = pred

        results.append((startX, startY, endX, endY, label, float(conf), color))

    return results


def draw_ui(frame, detections, fps, frame_count):
    """Draw medical-grade HUD over the frame."""
    overlay = frame.copy()

    # Top status bar
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 48), COLORS["overlay"], -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title
    cv2.putText(frame, "FACE MASK DETECTOR",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, COLORS["accent"], 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (frame.shape[1] - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Draw each detection
    mask_count    = 0
    no_mask_count = 0

    for (startX, startY, endX, endY, label, conf, color) in detections:
        if "✓" in label:
            mask_count += 1
        else:
            no_mask_count += 1

        # Box
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Label background
        text = f"{label} {conf*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lY = startY - 10 if startY - 10 > 20 else startY + th + 10
        cv2.rectangle(frame, (startX, lY - th - 4),
                      (startX + tw + 8, lY + 4), color, -1)
        cv2.putText(frame, text,
                    (startX + 4, lY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Bottom stats bar
    bar_y = frame.shape[0] - 40
    cv2.rectangle(frame, (0, bar_y - 5), (frame.shape[1], frame.shape[0]),
                  COLORS["overlay"], -1)
    cv2.putText(frame,
                f"Mask: {mask_count}  |  No Mask: {no_mask_count}  |  Faces: {len(detections)}",
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.putText(frame, "Press Q to quit",
                (frame.shape[1] - 150, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    return frame


# ─────────────────────────────────────────────
# MAIN INFERENCE LOOP
# ─────────────────────────────────────────────
def run_inference(source=0):
    import os

    download_face_model()
    face_net, mask_net = load_models()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Cannot open video source: {source}")
        return

    print("🎥 Live detection running... Press Q to quit.\n")

    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Stream ended.")
            break

        frame_count += 1

        # Compute FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time

        # Detect + classify
        detections = detect_and_classify(frame, face_net, mask_net)

        # Draw HUD
        frame = draw_ui(frame, detections, fps, frame_count)

        cv2.imshow("Face Mask Detector — Medical AI [QadeerDev]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Detection stopped.")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-Time Face Mask Detector"
    )
    parser.add_argument(
        "-s", "--source",
        type=str, default="0",
        help="Video source: 0 for webcam, or path to video file"
    )
    args = parser.parse_args()

    # Convert to int if webcam index
    source = int(args.source) if args.source.isdigit() else args.source
    run_inference(source)
