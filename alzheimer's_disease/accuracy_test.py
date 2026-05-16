import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_dir = r"E:\final_pro_alz\prepare_data_try1\test"

IMG_SIZE = (128, 128)
BATCH_SIZE = 16

model = tf.keras.models.load_model("alzheimer_new_4layer_model50_try.h5")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

loss, accuracy = model.evaluate(test_generator)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy * 100)
