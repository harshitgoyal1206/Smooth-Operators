import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from flask import Flask, request, render_template_string
import tensorflow as tf
import numpy as np
import cv2
import base64
from huggingface_hub import hf_hub_download

app = Flask(__name__)

print("TensorFlow version:", tf.__version__)

MODEL_PATH = hf_hub_download(
    repo_id="harshitgoyal1206/nt_model",
    filename="nt_model.keras"
)

print("Model downloaded at:", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Model loaded successfully")


def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (256,256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def analyze_image(image_path):

    original = cv2.imread(image_path)
    original = cv2.resize(original,(256,256))

    img = preprocess_image(image_path)

    pred = model.predict(img)[0,:,:,0]

    mask = pred > 0.35

    coords = np.where(mask)

    if len(coords[0]) == 0:
        thickness = 0
        top = bottom = 0
    else:
        top = coords[0].min()
        bottom = coords[0].max()
        thickness = bottom - top

    overlay = original.copy()

    if thickness > 0:
        cv2.line(overlay,(120,top),(120,bottom),(0,0,255),2)

    mask_img = (mask*255).astype(np.uint8)

    return original, mask_img, overlay, thickness


def image_to_base64(img):

    _, buffer = cv2.imencode(".png", img)
    encoded = base64.b64encode(buffer).decode()

    return encoded


@app.route("/")
def home():

    return """
    <html>
    <head>
    <title>Down Syndrome Detection</title>
    <style>
    body{
        font-family: Arial;
        text-align:center;
        background:#f2f2f2;
    }
    .card{
        background:white;
        padding:30px;
        margin:auto;
        width:500px;
        border-radius:10px;
        box-shadow:0 0 10px rgba(0,0,0,0.2);
    }
    button{
        padding:10px 20px;
        background:#007BFF;
        color:white;
        border:none;
        border-radius:5px;
    }
    </style>
    </head>

    <body>
    <div class="card">
    <h2>Down Syndrome Detection</h2>

    <form action="/predict" method="post" enctype="multipart/form-data">
    <input type="file" name="file" required><br><br>
    <button type="submit">Analyze Ultrasound</button>
    </form>

    </div>
    </body>
    </html>
    """


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    path = "temp.png"
    file.save(path)

    original, mask, overlay, nt_pixels = analyze_image(path)

    pixel_to_mm = 0.1
    nt_mm = nt_pixels * pixel_to_mm

    risk = "LOW RISK" if nt_mm < 3.5 else "HIGH RISK"

    original_b64 = image_to_base64(original)
    mask_b64 = image_to_base64(mask)
    overlay_b64 = image_to_base64(overlay)

    return render_template_string("""

    <html>
    <head>

    <style>

    body{
        font-family:Arial;
        text-align:center;
        background:#f5f5f5;
    }

    .container{
        width:900px;
        margin:auto;
    }

    img{
        width:250px;
        border-radius:10px;
        margin:10px;
        box-shadow:0 0 10px rgba(0,0,0,0.3);
    }

    .result{
        background:white;
        padding:20px;
        border-radius:10px;
        margin-top:20px;
        box-shadow:0 0 10px rgba(0,0,0,0.2);
    }

    </style>

    </head>

    <body>

    <div class="container">

    <h2>Down Syndrome Detection Result</h2>

    <div class="result">

    <h3>NT Thickness: {{nt}} mm</h3>
    <h3>Risk: {{risk}}</h3>

    </div>

    <h3>Images</h3>

    <img src="data:image/png;base64,{{orig}}">
    <img src="data:image/png;base64,{{mask}}">
    <img src="data:image/png;base64,{{overlay}}">

    <br><br>

    <a href="/">Analyze Another Image</a>

    </div>

    </body>

    </html>

    """,
    nt=round(nt_mm,2),
    risk=risk,
    orig=original_b64,
    mask=mask_b64,
    overlay=overlay_b64
    )


if __name__ == "__main__":

    port = int(os.environ.get("PORT",8080))

    app.run(host="0.0.0.0",port=port)
