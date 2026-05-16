from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing import image
app = Flask(__name__)
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_PATH = "alzheimer_new_4layer_model50_try.h5"
IMG_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 40.0
CLASSES = [
    'MildDemented',
    'ModerateDemented',
    'NonDemented',
    'VeryMildDemented'
]
DISPLAY_NAMES = {
    'MildDemented': "Mild Demented",
    'ModerateDemented': "Moderate Demented",
    'NonDemented': "Non-Demented",
    'VeryMildDemented': "Very Mild Demented"
}
model = tf.keras.models.load_model(MODEL_PATH)
def is_mri_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return False
    img = cv2.resize(img, IMG_SIZE)
    b, g, r = cv2.split(img)
    diff = (np.mean(abs(r-g)) + np.mean(abs(r-b)) + np.mean(abs(g-b))) / 3
    return diff < 25
def generate_heatmap(img_path):
    orig = cv2.imread(img_path)
    orig_resized = cv2.resize(orig, IMG_SIZE)
    img = orig_resized.astype("float32") / 255.
    img = np.expand_dims(img, axis=0)
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
            break
    feature_model = tf.keras.models.Model(inputs=model.inputs, outputs=last_conv.output)
    fmap = feature_model.predict(img)[0]
    fmap = np.maximum(fmap, 0)
    H, W, C = fmap.shape
    heatmap = np.zeros((H, W), dtype=np.float32)
    for i in range(C):
        fmap_norm = fmap[:, :, i] / (np.max(fmap[:, :, i]) + 1e-8)
        mask = cv2.resize(fmap_norm, IMG_SIZE)
        mimg = img.copy()
        mimg[0] *= mask[..., None]
        score = model.predict(mimg)[0].max()
        heatmap += score * fmap[:, :, i]
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heat = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, IMG_SIZE)), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_resized, 0.6, heat, 0.4, 0)
    threshold = np.percentile(heatmap, 85)
    mask = (heatmap >= threshold).astype(np.uint8)
    mask = cv2.resize(mask, IMG_SIZE) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(overlay, "Focus Region", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return overlay
def predict_stage(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img) / 255.
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr)[0]
    idx = np.argmax(pred)
    confidence = float(pred[idx] * 100)
    label = CLASSES[idx]
    if confidence < CONFIDENCE_THRESHOLD:
        return "Low Confidence", confidence
    return DISPLAY_NAMES[label], confidence
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    heatmap_path = None
    input_path = None
    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename != "":
            input_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
            file.save(input_path)
            if not is_mri_image(input_path):
                result = "Invalid image – Please upload a brain MRI scan."
                return render_template("index.html", result=result)
            result, confidence = predict_stage(input_path)
            overlay = generate_heatmap(input_path)
            heatmap_path = os.path.join(UPLOAD_FOLDER, "heatmap.jpg")
            cv2.imwrite(heatmap_path, overlay)
    return render_template("index.html",
                           result=result,
                           confidence=confidence,
                           input_path=input_path,
                           heatmap_path=heatmap_path)
if __name__ == "__main__":
    app.run(debug=True)
