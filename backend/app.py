"""Flask API for animal/bird detection using the custom YOLO model."""

from __future__ import annotations

import io
import os
from functools import lru_cache

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)
CORS(app)

# Maximum uploaded image size: 8 MB
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# Custom trained YOLO model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "best.pt",
)

# Minimum confidence for a detection
CONFIDENCE_THRESHOLD = 0.40


def allowed_file(filename: str) -> bool:
    """Return whether the filename has a supported image extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@lru_cache(maxsize=1)
def get_model():
    """Load the custom YOLO model once and reuse it."""
    from ultralytics import YOLO

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Custom model not found at: {MODEL_PATH}"
        )

    return YOLO(MODEL_PATH)


def classify(image: Image.Image) -> list[dict]:
    """
    Run the custom YOLO detector and return detected classes,
    confidence values, and bounding boxes.
    """
    model = get_model()

    # YOLO accepts PIL images directly.
    results = model.predict(
        source=image.convert("RGB"),
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    predictions: list[dict] = []

    if not results:
        return predictions

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return predictions

    names = result.names

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = xyxy

        predictions.append(
            {
                "label": str(names[class_id]),
                "confidence": round(confidence * 100, 2),
                "box": {
                    "x": round(x1, 2),
                    "y": round(y1, 2),
                    "width": round(x2 - x1, 2),
                    "height": round(y2 - y1, 2),
                },
            }
        )

    # Highest-confidence detections first
    predictions.sort(
        key=lambda item: item["confidence"],
        reverse=True,
    )

    # Return at most the top 5 detections
    return predictions[:5]


@app.get("/api/health")
def health():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "ok",
            "model": "custom YOLO",
            "model_path": MODEL_PATH,
        }
    )


@app.post("/api/classify")
def classify_image():
    """Accept an image upload and return YOLO detections."""
    uploaded = request.files.get("image")

    if uploaded is None or not uploaded.filename:
        return jsonify(
            {"error": "Choose an image to classify."}
        ), 400

    if not allowed_file(uploaded.filename):
        return jsonify(
            {"error": "Use a JPG, PNG, or WebP image."}
        ), 400

    try:
        contents = uploaded.read()

        if not contents:
            return jsonify(
                {"error": "The uploaded image is empty."}
            ), 400

        image = Image.open(io.BytesIO(contents))
        image.verify()

        # Re-open because verify() exhausts the image object.
        image = Image.open(io.BytesIO(contents)).convert("RGB")

    except (UnidentifiedImageError, OSError):
        return jsonify(
            {"error": "That file could not be read as an image."}
        ), 400

    try:
        predictions = classify(image)

    except FileNotFoundError as exc:
        app.logger.exception("Model file missing")
        return jsonify(
            {"error": str(exc)}
        ), 500

    except Exception as exc:
        app.logger.exception("Detection failed")
        return jsonify(
            {"error": f"Detection could not run: {exc}"}
        ), 503

    # No object passed the confidence threshold.
    if not predictions:
        return jsonify(
            {
                "predictions": [],
                "message": (
                    "No supported animal or bird was detected "
                    f"with confidence >= "
                    f"{CONFIDENCE_THRESHOLD * 100:.0f}%."
                ),
            }
        )

    return jsonify(
        {
            "predictions": predictions,
        }
    )


@app.errorhandler(413)
def file_too_large(_error):
    """Handle images larger than 8 MB."""
    return jsonify(
        {"error": "Image must be 8 MB or smaller."}
    ), 413


if __name__ == "__main__":
    app.run(debug=True, port=5000)