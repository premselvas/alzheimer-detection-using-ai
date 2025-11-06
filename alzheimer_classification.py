import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

train_dir = "dataset_combined/train"
test_dir = "dataset_combined/test"

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 3  

def build_fast_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.9  
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)
model = build_fast_cnn((IMG_SIZE[0], IMG_SIZE[1], 3), num_classes)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("🚀 Starting quick training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator
)

loss, acc = model.evaluate(test_generator)
print(f"✅ Quick Test Accuracy: {acc*100:.2f}%")

model.save("alzheimer_quick_model.h5")
print("💾 Model saved as 'alzheimer_quick_model.h5'")
