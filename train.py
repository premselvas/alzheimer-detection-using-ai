
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import pathlib

DATA_DIR = r"C:\Users\admin\Desktop\new alzheimer 1\dataset_combined"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")
test_dir = os.path.join(DATA_DIR, "test")
NUM_CLASSES = len(os.listdir(train_dir))

def check_images(path):
    bad_files = []
    path = pathlib.Path(path)
    for img_path in path.rglob("*.jpg"):
        try:
            img = Image.open(img_path)
            img.verify()
        except Exception:
            bad_files.append(str(img_path))
    return bad_files

print("🔍 Checking dataset integrity...")
bad_images = check_images(DATA_DIR)
if bad_images:
    print("⚠️ Found corrupted images:")
    for f in bad_images:
        print("  ", f)
else:
    print("✅ All images are valid.")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("✅ Data generators created successfully.")

y_train = train_gen.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: float(w) for i, w in enumerate(class_weights)}

print("Class indices:", train_gen.class_indices)
print("Class weights:", class_weights)

base_model = EfficientNetB0(
    weights=None,  
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base_model.trainable = False  

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

os.makedirs("models", exist_ok=True)
checkpoint_cb = ModelCheckpoint("models/best_model.keras", save_best_only=True, monitor="val_loss", mode="min")
earlystop_cb = EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss")
reduce_lr_cb = ReduceLROnPlateau(patience=3, factor=0.5, verbose=1)

print("\n🚀 Training base model...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
)


print("\n🔧 Fine-tuning last 50 layers...")
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    class_weight=class_weights,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
)

model.save("models/final_model.h5")
print("✅ Training complete! Model saved in 'models/' folder.")
