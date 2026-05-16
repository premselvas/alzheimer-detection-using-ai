import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = r"E:\final_pro_alz\prepare_data_try1\train"

IMG_SIZE = (128, 128)
BATCH_SIZE = 1

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=False
)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

model = tf.keras.models.load_model("alzheimer_new_4layer_model50_try.h5")

train_loss, train_acc = model.evaluate(train_generator, verbose=1)
val_loss, val_acc = model.evaluate(val_generator, verbose=1)

print(f"Training Accuracy     : {train_acc * 100:.2f}%")
print(f"Validation Accuracy   : {val_acc * 100:.2f}%")
