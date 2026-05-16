import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import numpy as np

MODEL_PATH = r"E:\final_pro_alz\alzheimer_new_4layer_model22.h5"

TRAIN_DIR = r"E:\final_pro_alz\prepare_data_try1\train"
VAL_DIR   = r"E:\final_pro_alz\prepare_data_try1\val"

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10

QUICK_CHECK = True
QUICK_TRAIN_STEPS = 5
QUICK_VAL_STEPS = 5
QUICK_EPOCHS = 3

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

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
model.summary()

if QUICK_CHECK:
    print(f"QUICK_CHECK enabled: running {QUICK_EPOCHS} mini-epochs, {QUICK_TRAIN_STEPS} train steps/epoch and {QUICK_VAL_STEPS} val steps/epoch")

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    names = getattr(model, 'metrics_names', None) or []
    acc_idx = None
    for candidate in ('accuracy', 'acc'):
        if candidate in names:
            acc_idx = names.index(candidate)
            break

    for e in range(QUICK_EPOCHS):
        
        train_metrics_accum = None
        for step in range(QUICK_TRAIN_STEPS):
            try:
                x_batch, y_batch = next(train_generator)
            except Exception:
                
                train_generator = train_datagen.flow_from_directory(
                    TRAIN_DIR,
                    target_size=IMG_SIZE,
                    batch_size=BATCH_SIZE,
                    class_mode='categorical'
                )
                x_batch, y_batch = next(train_generator)

            metrics = model.train_on_batch(x_batch, y_batch)
            if train_metrics_accum is None:
                train_metrics_accum = metrics
            else:
                if isinstance(metrics, (list, tuple)):
                    train_metrics_accum = [a + b for a, b in zip(train_metrics_accum, metrics)]
                else:
                    train_metrics_accum = train_metrics_accum + metrics

       
        if isinstance(train_metrics_accum, (list, tuple)):
            train_metrics_avg = [v / QUICK_TRAIN_STEPS for v in train_metrics_accum]
        else:
            train_metrics_avg = train_metrics_accum / QUICK_TRAIN_STEPS

      
        val_metrics = model.evaluate(val_generator, steps=QUICK_VAL_STEPS, verbose=0)

        if isinstance(train_metrics_avg, (list, tuple)):
            train_loss_list.append(float(train_metrics_avg[0]))
            if acc_idx is not None and len(train_metrics_avg) > acc_idx:
                train_acc_list.append(float(train_metrics_avg[acc_idx]))
        else:
            train_loss_list.append(float(train_metrics_avg))

        if isinstance(val_metrics, (list, tuple)):
            val_loss_list.append(float(val_metrics[0]))
            if acc_idx is not None and len(val_metrics) > acc_idx:
                val_acc_list.append(float(val_metrics[acc_idx]))
        else:
            val_loss_list.append(float(val_metrics))

    class H:
        pass

    history = H()
    history.history = {}
    history.history['loss'] = train_loss_list
    history.history['val_loss'] = val_loss_list
    history.history['accuracy'] = train_acc_list if train_acc_list else None
    history.history['val_accuracy'] = val_acc_list if val_acc_list else None
else:
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )

plt.style.use('seaborn-darkgrid')

def smooth_curve(x, window=3):
    if window <= 1:
        return np.array(x)
    w = np.ones(window) / window
    return np.convolve(x, w, mode='same')


def plot_with_smoothing(train_vals, val_vals, ylabel, title, filename, smooth_window=3):
    if not train_vals or not val_vals:
        print(f'Not enough data to plot {title}.')
        return

    n = len(train_vals)
    x = np.arange(n)  

    sw = smooth_window if n >= smooth_window * 2 else 1
    train_smooth = smooth_curve(train_vals, sw) if sw > 1 else np.array(train_vals)
    val_smooth = smooth_curve(val_vals, sw) if sw > 1 else np.array(val_vals)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, train_vals, color='#1f77b4', linewidth=1.5, alpha=0.6, label='Training')
    ax.plot(x, val_vals, color='#ff7f0e', linewidth=1.5, alpha=0.6, label='Validation')
    if sw > 1:
        ax.plot(x, train_smooth, color='#1f77b4', linewidth=2.0, linestyle='-', label=f'Training (smoothed w={sw})')
        ax.plot(x, val_smooth, color='#ff7f0e', linewidth=2.0, linestyle='-', label=f'Validation (smoothed w={sw})')

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax.set_xlim(0, max(1, n - 1))

    if ylabel.lower().startswith('acc') or ylabel.lower().startswith('accuracy'):
        ax.set_ylim(0.0, 1.02)
    else:
        ymin = 0.0
        ymax = max(max(train_vals), max(val_vals)) * 1.05
        ax.set_ylim(ymin, ymax)

    if n <= 20:
        ax.set_xticks(x)
    else:
        ax.set_xticks(np.linspace(0, n - 1, min(11, n)).astype(int))

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


loss = history.history.get('loss', [])
val_loss = history.history.get('val_loss', [])
acc = history.history.get('accuracy') or history.history.get('acc')
val_acc = history.history.get('val_accuracy') or history.history.get('val_acc')

plot_with_smoothing(acc if acc else [], val_acc if val_acc else [], 'Accuracy', 'Training and validation Accuracy', 'accuracy_graph.png', smooth_window=5)
plot_with_smoothing(loss if loss else [], val_loss if val_loss else [], 'Loss', 'Training and validation Loss', 'loss_graph.png', smooth_window=5)

print("Graphs saved successfully!")

def detect_overfitting(history, patience=3):
    """Return a short diagnosis string about overfitting.

    Heuristic: for the last `patience` epochs, if training loss decreased
    each epoch while validation loss increased each epoch, we flag overfitting.
    Similarly, if training accuracy increased while validation accuracy
    decreased for `patience` epochs, we flag overfitting.
    """
    h = history.history
    losses = h.get('loss')
    val_losses = h.get('val_loss')
    acc = h.get('accuracy') or h.get('acc')
    val_acc = h.get('val_accuracy') or h.get('val_acc')

    if not losses or not val_losses:
        return "Insufficient loss history to decide."

    n = len(losses)
    if n < patience + 1:
        return f"Not enough epochs ({n}) to apply heuristic (need > {patience})."

    loss_pattern = all(losses[-i] < losses[-i-1] for i in range(1, patience+1))
    val_loss_pattern = all(val_losses[-i] > val_losses[-i-1] for i in range(1, patience+1))

    acc_pattern = False
    if acc is not None and val_acc is not None:
        acc_pattern = all(acc[-i] > acc[-i-1] for i in range(1, patience+1)) and all(val_acc[-i] < val_acc[-i-1] for i in range(1, patience+1))

    if loss_pattern and val_loss_pattern:
        return "Likely overfitting: training loss decreasing while validation loss increasing."
    if acc_pattern:
        return "Likely overfitting: training accuracy increasing while validation accuracy decreasing."

    return "No clear overfitting detected by the simple heuristic."


diagnosis = detect_overfitting(history, patience=3)
print("Overfitting check:", diagnosis)

report_lines = [
    f"Diagnosis: {diagnosis}",
    f"Epochs: {len(loss) if loss else 'N/A'}",
]
if loss:
    report_lines.append(f"Final training loss: {loss[-1]:.4f}")
if val_loss:
    report_lines.append(f"Final validation loss: {val_loss[-1]:.4f}")
if acc is not None:
    report_lines.append(f"Final training accuracy: {acc[-1]:.4f}")
if val_acc is not None:
    report_lines.append(f"Final validation accuracy: {val_acc[-1]:.4f}")

with open('overfit_report.txt', 'w') as f:
    f.write('\n'.join(report_lines))

print('Saved overfit_report.txt')