import gradio as gr
import cv2
import numpy as np
import tensorflow as tf

# Custom functions
def dice_coef(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.*intersection + 1)/(tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

bce = tf.keras.losses.BinaryCrossentropy()

def combined_loss(y_true, y_pred):
    return bce(y_true, y_pred) + dice_loss(y_true, y_pred)

# Load your trained model
model = tf.keras.models.load_model(
    "nt_model.keras",
    custom_objects={"combined_loss": combined_loss, "dice_coef": dice_coef}
)

# Prediction function
def predict(image):
    img = cv2.resize(image,(256,256))
    img = img/255.0
    img = np.expand_dims(img,0)
    pred = model.predict(img)[0,:,:,0]
    mask = pred > 0.35
    coords = np.where(mask)
    if len(coords[0])==0:
        nt_pixels = 0
    else:
        nt_pixels = coords[0].max() - coords[0].min()
    nt_mm = nt_pixels * 0.1
    risk = "LOW RISK" if nt_mm < 3.5 else "HIGH RISK"
    return f"NT Thickness: {nt_mm:.2f} mm | Risk: {risk}"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Fetal NT Thickness Detection"
)

demo.launch()