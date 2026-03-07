from flask import Flask, request
import tensorflow as tf
import cv2
import numpy as np
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Download model from Hugging Face
MODEL_PATH = hf_hub_download(
    repo_id="harshitgoyal1206/nt_model",
    filename="nt_model.keras"
)

# Load model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


def predict_nt(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, 0)

    pred = model.predict(img)[0, :, :, 0]
    mask = pred > 0.35

    coords = np.where(mask)

    if len(coords[0]) == 0:
        return 0

    thickness = coords[0].max() - coords[0].min()

    return thickness


def classify_risk(image_path):

    nt_pixels = predict_nt(image_path)

    pixel_to_mm = 0.1
    nt_mm = nt_pixels * pixel_to_mm

    if nt_mm < 3.5:
        risk = "LOW RISK"
    else:
        risk = "HIGH RISK"

    return nt_mm, risk


@app.route("/")
def home():

    return """
    <h2>Down Syndrome Detection</h2>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit">
    </form>
    """


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    path = "temp.png"
    file.save(path)

    nt_mm, risk = classify_risk(path)

    return f"""
    NT thickness: {nt_mm:.2f} mm <br>
    Risk: {risk}
    """


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
