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

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Model loaded successfully")


def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(256,256))
    img = img/255.0
    img = np.expand_dims(img,axis=0)
    return img


def analyze_image(path):

    original = cv2.imread(path)
    original = cv2.resize(original,(256,256))

    img = preprocess_image(path)

    pred = model.predict(img)[0,:,:,0]

    mask = pred > 0.35

    coords = np.where(mask)

    if len(coords[0]) == 0:
        thickness = 0
        top = bottom = 0
    else:
        top = coords[0].min()
        bottom = coords[0].max()
        thickness = bottom-top

    overlay = original.copy()

    if thickness>0:
        cv2.line(overlay,(120,top),(120,bottom),(0,0,255),2)

    mask_img = (mask*255).astype(np.uint8)

    return original,mask_img,overlay,thickness


def encode_image(img):

    _,buffer = cv2.imencode(".png",img)
    return base64.b64encode(buffer).decode()


@app.route("/")
def home():

    return """

<html>

<head>

<title>AI Down Syndrome Detection</title>

<style>

body{
font-family:Arial;
background:#eef2f7;
text-align:center;
}

.container{
width:700px;
margin:auto;
margin-top:80px;
}

.card{
background:white;
padding:40px;
border-radius:12px;
box-shadow:0 8px 20px rgba(0,0,0,0.15);
}

h1{
color:#2c3e50;
}

.upload-box{
border:2px dashed #3498db;
padding:40px;
margin-top:20px;
border-radius:10px;
background:#f9fbff;
}

button{
margin-top:20px;
padding:12px 25px;
background:#3498db;
color:white;
border:none;
border-radius:6px;
font-size:16px;
cursor:pointer;
}

button:hover{
background:#2980b9;
}

</style>

</head>

<body>

<div class="container">

<div class="card">

<h1>AI Nuchal Translucency Analysis</h1>

<p>Upload fetal ultrasound for automated Down Syndrome risk analysis</p>

<form action="/predict" method="post" enctype="multipart/form-data">

<div class="upload-box">
<input type="file" name="file" required>
</div>

<button type="submit">Analyze Ultrasound</button>

</form>

</div>

</div>

</body>

</html>

"""


@app.route("/predict",methods=["POST"])
def predict():

    file = request.files["file"]

    path="temp.png"
    file.save(path)

    original,mask,overlay,nt_pixels = analyze_image(path)

    pixel_to_mm = 0.1
    nt_mm = nt_pixels*pixel_to_mm

    risk = "LOW RISK" if nt_mm<3.5 else "HIGH RISK"

    color = "green" if risk=="LOW RISK" else "red"

    orig_b64 = encode_image(original)
    mask_b64 = encode_image(mask)
    overlay_b64 = encode_image(overlay)

    return render_template_string("""

<html>

<head>

<title>Analysis Result</title>

<style>

body{
font-family:Arial;
background:#eef2f7;
text-align:center;
}

.container{
width:1000px;
margin:auto;
margin-top:40px;
}

.card{
background:white;
padding:30px;
border-radius:12px;
box-shadow:0 8px 20px rgba(0,0,0,0.15);
}

.images{
display:flex;
justify-content:center;
gap:30px;
margin-top:20px;
}

img{
width:280px;
border-radius:10px;
box-shadow:0 6px 15px rgba(0,0,0,0.2);
}

.result{
font-size:20px;
margin-top:10px;
}

.risk{
font-size:24px;
font-weight:bold;
color:{{color}};
}

button{
margin-top:20px;
padding:10px 20px;
background:#3498db;
border:none;
color:white;
border-radius:6px;
cursor:pointer;
}

button:hover{
background:#2980b9;
}

</style>

</head>

<body>

<div class="container">

<div class="card">

<h2>Analysis Result</h2>

<div class="result">
NT Thickness: <b>{{nt}} mm</b>
</div>

<div class="risk">
{{risk}}
</div>

<div class="images">

<div>
<p>Original</p>
<img src="data:image/png;base64,{{orig}}">
</div>

<div>
<p>Segmentation Mask</p>
<img src="data:image/png;base64,{{mask}}">
</div>

<div>
<p>Measurement Overlay</p>
<img src="data:image/png;base64,{{overlay}}">
</div>

</div>

<br>

<a href="/"><button>Analyze Another Image</button></a>

</div>

</div>

</body>

</html>

""",
nt=round(nt_mm,2),
risk=risk,
color=color,
orig=orig_b64,
mask=mask_b64,
overlay=overlay_b64
)


if __name__=="__main__":

    port=int(os.environ.get("PORT",8080))

    app.run(host="0.0.0.0",port=port)
