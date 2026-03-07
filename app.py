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
background:linear-gradient(135deg,#e3f2fd,#f5f7fa);
text-align:center;
}

.container{
width:750px;
margin:auto;
margin-top:80px;
}

.card{
background:white;
padding:40px;
border-radius:14px;
box-shadow:0 10px 30px rgba(0,0,0,0.15);
}

.upload-box{
border:2px dashed #1976d2;
padding:30px;
border-radius:10px;
background:#f8fbff;
}

.preview img{
margin-top:20px;
width:260px;
border-radius:8px;
box-shadow:0 6px 15px rgba(0,0,0,0.2);
}

button{
margin-top:20px;
padding:12px 30px;
background:#1976d2;
color:white;
border:none;
border-radius:8px;
font-size:16px;
cursor:pointer;
}

button:hover{
background:#0d47a1;
}

#loading{
display:none;
margin-top:20px;
}

.spinner{
border:6px solid #f3f3f3;
border-top:6px solid #1976d2;
border-radius:50%;
width:40px;
height:40px;
animation:spin 1s linear infinite;
margin:auto;
}

@keyframes spin{
0%{transform:rotate(0deg);}
100%{transform:rotate(360deg);}
}

</style>

<script>

function previewImage(event){

var reader=new FileReader();

reader.onload=function(){
var img=document.getElementById("preview");
img.src=reader.result;
img.style.display="block";
}

reader.readAsDataURL(event.target.files[0]);

}

function showLoading(){
document.getElementById("loading").style.display="block";
}

</script>

</head>

<body>

<div class="container">

<div class="card">

<h1>AI Nuchal Translucency Analysis</h1>

<p>Upload fetal ultrasound to estimate NT thickness and Down Syndrome risk</p>

<form action="/predict" method="post" enctype="multipart/form-data" onsubmit="showLoading()">

<div class="upload-box">

<input type="file" name="file" accept="image/*" onchange="previewImage(event)" required>

</div>

<div class="preview">
<img id="preview" style="display:none">
</div>

<button type="submit">Run AI Analysis</button>

<div id="loading">

<div class="spinner"></div>

<p>Processing image with AI model...</p>

</div>

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
background:linear-gradient(135deg,#e3f2fd,#f5f7fa);
text-align:center;
}

.container{
width:1100px;
margin:auto;
margin-top:40px;
}

.card{
background:white;
padding:30px;
border-radius:14px;
box-shadow:0 10px 30px rgba(0,0,0,0.15);
}

.images{
display:flex;
justify-content:center;
gap:30px;
margin-top:20px;
}

img{
width:300px;
border-radius:10px;
box-shadow:0 6px 20px rgba(0,0,0,0.2);
}

.result-box{
background:#f9fbff;
padding:20px;
border-radius:10px;
margin-top:10px;
}

.metric{
font-size:20px;
margin:10px;
}

.risk{
font-size:28px;
font-weight:bold;
color:{{color}};
}

.info{
margin-top:20px;
font-size:15px;
color:#555;
line-height:1.6;
}

button{
margin-top:20px;
padding:10px 20px;
background:#1976d2;
border:none;
color:white;
border-radius:8px;
cursor:pointer;
}

button:hover{
background:#0d47a1;
}

</style>

</head>

<body>

<div class="container">

<div class="card">

<h2>AI Ultrasound Analysis Result</h2>

<div class="result-box">

<div class="metric">
Nuchal Translucency Thickness: <b>{{nt}} mm</b>
</div>

<div class="risk">
Risk Classification: {{risk}}
</div>

<div class="metric">
Clinical Threshold: 3.5 mm
</div>

</div>

<h3>Model Visualizations</h3>

<div class="images">

<div>
<p>Original Ultrasound</p>
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

<div class="info">

<b>Explanation:</b><br><br>

• The deep learning model segments the Nuchal Translucency region.<br>
• Thickness is computed from the vertical span of the segmented mask.<br>
• NT > 3.5 mm is associated with increased Down Syndrome risk.<br>
• This system assists screening and is not a medical diagnosis.

</div>

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
