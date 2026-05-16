import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

MODEL_PATH = r"E:\final_pro_alz\alzheimer_quick_model.h5"
TRAIN_DIR  = r"E:\final_pro_alz\prepare_data_try1\train"
VAL_DIR    = r"E:\final_pro_alz\prepare_data_try1\val"

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 3  # MUST be > 1

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

model = load_model(MODEL_PATH)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy of Hybrid CNN')
plt.legend()
plt.grid(True)

plt.savefig("accuracy_graph.png", dpi=300, bbox_inches="tight")
plt.show()

print("Accuracy graph saved successfully!")
