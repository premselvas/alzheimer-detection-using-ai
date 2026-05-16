import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score
import numpy as np
test_dir = r"E:\final_pro_alz\prepare_data_try1\test"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
model = tf.keras.models.load_model("alzheimer_new_4layer_model22.h5")
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
y_pred_prob = model.predict(test_generator)
y_true = test_generator.classes
y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=len(test_generator.class_indices))
auc = roc_auc_score(y_true_one_hot, y_pred_prob, multi_class='ovr')
print("Test AUC:", auc)