import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model_path = r"E:\final_pro_alz\alzheimer_new_4layer_model22.h5"  
img_path = r"E:\final_pro_alz\for_teesting_img\00c3e349-8f34-4688-aa36-18b9bfde54c1.jpg"  

model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

IMG_SIZE = (128, 128)  
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = img_array / 255.0  
img_array = np.expand_dims(img_array, axis=0)  

pred = model.predict(img_array)
pred_class = class_names[np.argmax(pred)]
pred_prob = np.max(pred)

print(f"Predicted class: {pred_class}")
print(f"Confidence Level: {pred_prob:.4f}")