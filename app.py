from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# -------------------------
# Download model from HF
# -------------------------

MODEL_PATH = hf_hub_download(
    repo_id="harshitgoyal1206/nt_model",
    filename="nt_model.keras"
)

# -------------------------
# Load model once
# -------------------------

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Model loaded successfully")


# -------------------------
# Image preprocessing
# -------------------------

def preprocess_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Invalid image")

    img = cv2.resize(img, (256, 256))
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img


# -------------------------
# NT prediction
# -------------------------

def predict_nt(image_path):

    img = preprocess_image(image_path)

    prediction = model.predict(img)

    mask = prediction[0, :, :, 0] > 0.35

    coords = np.where(mask)

    if len(coords[0]) == 0:
        return 0

    thickness_pixels = coords[0].max() - coords[0].min()

    return thickness_pixels


# -------------------------
# Risk classification
# -------------------------

def classify_risk(image_path):

    nt_pixels = predict_nt(image_path)

    pixel_to_mm = 0.1
    nt_mm = nt_pixels * pixel_to_mm

    if nt_mm < 3.5:
        risk = "LOW RISK"
    else:
        risk = "HIGH RISK"

    return nt_mm, risk


# -------------------------
# Routes
# -------------------------

@app.route("/")
def home():

    return """
    <h2>Down Syndrome Detection (NT Measurement)</h2>
    <form action="/predict" method="post" enctype="multipart/form-data">
        Upload Ultrasound Image:<br><br>
        <input type="file" name="file"><br><br>
        <input type="submit">
    </form>
    """


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"})

    temp_path = "temp_image.png"
    file.save(temp_path)

    try:

        nt_mm, risk = classify_risk(temp_path)

        result = {
            "nt_thickness_mm": round(float(nt_mm), 2),
            "risk": risk
        }

    except Exception as e:

        result = {"error": str(e)}

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return jsonify(result)


# -------------------------
# Run server
# -------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
