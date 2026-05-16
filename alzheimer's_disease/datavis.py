import os
import random
import matplotlib.pyplot as plt
from PIL import Image

DATASET_PATH = r"E:\final_pro_alz\prepare_data_try1\train"
CLASSES = ["ModerateDemented", "NonDemented", "VeryMildDemented", "MildDemented"]
SAMPLES_PER_CLASS = 5  
IMG_SIZE = (128, 128)

fig, axes = plt.subplots(len(CLASSES), SAMPLES_PER_CLASS, figsize=(10, 7))

for row, cls in enumerate(CLASSES):
    class_dir = os.path.join(DATASET_PATH, cls)
    images = os.listdir(class_dir)
    random.shuffle(images)

    for col in range(SAMPLES_PER_CLASS):
        img_path = os.path.join(class_dir, images[col])
        img = Image.open(img_path).resize(IMG_SIZE)
        axes[row, col].imshow(img, cmap="gray")
        axes[row, col].axis("off")

        if col == 0:
            axes[row, col].set_ylabel(cls.replace("Demented", " Demented"), size=12)

plt.tight_layout()
plt.show()