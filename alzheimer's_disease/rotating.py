import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os

img_path = r"E:\final_pro_alz\for_teesting_img\0a4e9ed1-fbca-4d68-9cb9-54db8acffd56.jpg"
img = load_img(img_path, target_size=(128, 128))

img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2]
)


augmented_images = datagen.flow(img_array, batch_size=1)

plt.figure(figsize=(10, 6))

plt.subplot(2, 4, 1)
plt.imshow(img_array[0].astype("uint8"))
plt.title("Original")
plt.axis("off")
for i in range(7):
    plt.subplot(2, 4, i + 2)
    aug_img = next(augmented_images)[0].astype("uint8")
    plt.imshow(aug_img)
    plt.title(f"Aug {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()
