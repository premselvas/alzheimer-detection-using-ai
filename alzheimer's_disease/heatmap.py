import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os

model_path = r"E:\final_pro_alz\alzheimer_new_4layer_model30.h5"

if not os.path.exists(model_path):
    print("ERROR: Model file not found")
    print("Check this path:", model_path)
    exit()
else:
    print("Model found")

model = tf.keras.models.load_model(model_path)

img_path = r"E:\final_pro_alz\for_teesting_img\0a4e9ed1-fbca-4d68-9cb9-54db8acffd56.jpg"

if not os.path.exists(img_path):
    print("ERROR: Image not found")
    exit()

img = cv2.imread(img_path)
img = cv2.resize(img, (128, 128))

img_array = img / 255.0
img_array = np.expand_dims(img_array, axis=0)

last_conv_layer = None
for layer in reversed(model.layers):
    if "conv" in layer.name.lower():
        last_conv_layer = layer.name
        break

if last_conv_layer is None:
    print("ERROR: No Conv layer found")
    exit()

print("Last Conv Layer:", last_conv_layer)

grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(last_conv_layer).output, model.output]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    class_index = np.argmax(predictions[0])
    loss = predictions[:, class_index]

grads = tape.gradient(loss, conv_outputs)

if grads is None:
    print("ERROR: Gradient issue")
    exit()

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

heatmap = np.maximum(heatmap, 0)
heatmap = heatmap / (np.max(heatmap) + 1e-8)

heatmap = cv2.resize(heatmap.numpy(), (128, 128))

heatmap_color = cv2.applyColorMap(
    np.uint8(255 * heatmap), cv2.COLORMAP_JET
)

overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

_, thresh = cv2.threshold(
    np.uint8(255 * heatmap), 150, 255, cv2.THRESH_BINARY
)

contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 10 and h > 10:
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Heatmap + Bounding Box")
plt.axis("off")
plt.show()

output_path = r"E:\final_pro_alz\output.jpg"
cv2.imwrite(output_path, overlay)

print("Output saved at:", output_path)