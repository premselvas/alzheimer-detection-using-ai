import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("alzheimer_quick_model.h5")

class_labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

img_path = r"C:\Users\admin\Desktop\new alzheimer 1\sample images\ModerateDemented_199.jpg"

img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

predictions = model.predict(img_array)
predicted_class = class_labels[np.argmax(predictions)]

print(f"🧠 Predicted Class: {predicted_class}")
