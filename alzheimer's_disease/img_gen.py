import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

train_dir = r"E:\final_pro_alz\prepare_data_try1\train"

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Train directory not found: {train_dir}")

classes = [d for d in os.listdir(train_dir)
           if os.path.isdir(os.path.join(train_dir, d))]

if len(classes) < 2:
    raise ValueError("At least two class folders are required.")

print("Detected class folders:", classes)

positive_dir = os.path.join(train_dir, classes[0])
negative_dir = os.path.join(train_dir, classes[1])

pos_images = os.listdir(positive_dir)
neg_images = os.listdir(negative_dir)

if len(pos_images) == 0 or len(neg_images) == 0:
    raise ValueError("One of the class folders is empty.")

pos_img_name = random.choice(pos_images)
neg_img_name = random.choice(neg_images)

pos_img_path = os.path.join(positive_dir, pos_img_name)
neg_img_path = os.path.join(negative_dir, neg_img_name)

pos_img = image.load_img(pos_img_path, target_size=(128, 128))
neg_img = image.load_img(neg_img_path, target_size=(128, 128))

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(pos_img)
plt.title("Figure 2: Positive MRI Alzheimer's")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(neg_img)
plt.title("Figure 3: Negative MRI Alzheimer's")
plt.axis("off")

plt.tight_layout()

plt.savefig("Figure_2_3_MRI_Samples.png", dpi=300, bbox_inches="tight")

plt.show()

print("Figures generated and saved successfully.")
