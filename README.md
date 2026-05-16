# alzheimer-detection-using-ai
# 🧠 Alzheimer's Disease Detection System using Deep Learning

An advanced AI-powered web application for detecting and classifying Alzheimer's disease stages from brain MRI scans using Convolutional Neural Networks (CNN). This project provides accurate multi-class prediction along with visual heatmap explanations to assist in understanding model decisions.

---

# 📌 Project Overview

Alzheimer’s disease is a progressive neurological disorder that affects memory, thinking, and behavior. Early detection plays a critical role in improving patient care and treatment planning.

This project uses Deep Learning and Computer Vision techniques to analyze MRI brain scans and classify them into different stages of Alzheimer’s disease.

The system includes:

- 🧠 CNN-based MRI classification model
- 🌐 Flask-based web application
- 🔥 Heatmap visualization using Score-CAM
- 📊 Accuracy, AUC, and overfitting analysis
- ✅ MRI image validation
- 📈 Confidence score prediction

---

# 🚀 Features

| Feature | Description |
|---|---|
| 🧠 Multi-Class Classification | Predicts 4 stages of Alzheimer's disease |
| 🔥 Heatmap Visualization | Score-CAM highlights important MRI regions |
| 🌐 Flask Web Interface | Upload MRI images through browser |
| 📊 Confidence Score | Shows prediction probability |
| ✅ MRI Validation | Checks whether uploaded image is a brain MRI |
| 📉 Overfitting Detection | Training vs validation performance monitoring |
| 📈 Graph Visualization | Accuracy and loss graphs generation |

---

# 📂 Alzheimer's Stages Classified

| Class | Description |
|---|---|
| **Non-Demented** | Healthy brain MRI |
| **Very Mild Demented** | Early-stage Alzheimer's |
| **Mild Demented** | Moderate cognitive decline |
| **Moderate Demented** | Advanced Alzheimer's stage |

---

# 🏗️ Model Architecture

The project uses a custom-built **4-layer CNN architecture**.

```text
Input Layer (128x128x3)
        ↓
Conv2D (16 Filters) + MaxPooling
        ↓
Conv2D (32 Filters) + MaxPooling
        ↓
Conv2D (64 Filters) + MaxPooling
        ↓
Conv2D (128 Filters) + MaxPooling
        ↓
Flatten Layer
        ↓
Dense (128, ReLU)
        ↓
Dropout (0.3)
        ↓
Dense (4, Softmax)
```

---

# ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Learning Rate | 0.001 |
| Batch Size | 16 |
| Epochs | 50 |
| Image Size | 128×128 |

---

# 📊 Dataset Information

Dataset used: **Alzheimer MRI Dataset from Kaggle**

The dataset contains:
- Original MRI Images
- Augmented MRI Images

---

# 📁 Dataset Split

| Dataset | Percentage |
|---|---|
| Training | 80% |
| Validation | 10% |
| Testing | 10% |

---

# 🖼️ Image Preprocessing

The following preprocessing techniques were applied:

- Resize images to `128×128`
- Normalize pixel values to `[0,1]`
- Data augmentation techniques:
  - Rotation (`±15°`)
  - Zoom (`0.1`)
  - Horizontal Flip
  - Brightness Adjustment (`0.8 - 1.2`)

---

# 📁 Project Structure

```text
Alzheimers-Disease-Detection/
│
├── app.py                     # Flask web application
├── train_classi.py            # CNN model training
├── data_split.py              # Dataset splitting
├── img_classi.py              # Single image prediction
├── heatmap.py                 # Score-CAM heatmap generation
├── accuracy_test.py           # Accuracy evaluation
├── auc_test.py                # ROC-AUC calculation
├── overfitgra.py              # Overfitting analysis
├── overfitting_test.py        # Generalization testing
├── accuracygraph.py           # Accuracy graph plotting
├── graph.py                   # Loss graph plotting
├── datavis.py                 # Dataset visualization
├── rotating.py                # Data augmentation demo
├── img_gen.py                 # Sample image generation
│
├── templates/
│   └── index.html             # Frontend webpage
│
├── static/
│   ├── uploads/               # Uploaded MRI images
│   └── heatmaps/              # Generated heatmaps
│
└── README.md
```

---

# 📈 Evaluation Metrics

The model performance is evaluated using:

| Metric | Purpose |
|---|---|
| Accuracy | Overall prediction performance |
| Loss | Training error measurement |
| ROC-AUC | Multi-class classification quality |
| Confidence Score | Prediction certainty |

---

# 🖥️ Installation & Setup

## 🔹 Prerequisites

Make sure Python 3.8+ is installed.

Required libraries:
- TensorFlow
- Flask
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Pillow

---

# 📥 Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/alzheimers-detection.git

cd alzheimers-detection
```

---

# 📦 Step 2: Install Dependencies

```bash
pip install tensorflow flask opencv-python numpy matplotlib scikit-learn pillow
```

---

# 📊 Step 3: Prepare Dataset

Update dataset paths inside `data_split.py`

```python
orig_dir = r"path/to/OriginalDataset"
aug_dir  = r"path/to/AugmentedAlzheimerDataset"
out_root = r"path/to/prepare_data_try1"
```

Run dataset preparation:

```bash
python data_split.py
```

---

# 🧠 Step 4: Train the CNN Model

```bash
python train_classi.py
```

---

# 🌐 Step 5: Run Flask Application

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

# 🧪 Usage Examples

## 🔹 Predict Single MRI Image

```bash
python img_classi.py
```

---

## 🔹 Generate Heatmap

```bash
python heatmap.py
```

---

## 🔹 Evaluate Accuracy

```bash
python accuracy_test.py
```

---

## 🔹 Calculate AUC Score

```bash
python auc_test.py
```

---

## 🔹 Check Overfitting

```bash
python overfitgra.py
```

---

## 🔹 Visualize Dataset

```bash
python datavis.py
```

---

# 🔥 Heatmap Visualization

The system uses **Score-CAM** visualization to identify important regions of the MRI image responsible for the prediction.

Benefits:
- Improves explainability
- Helps understand CNN decisions
- Highlights affected brain regions

---

# 📉 Overfitting Analysis

The project includes:
- Accuracy comparison graphs
- Loss comparison graphs
- Validation monitoring
- Generalization checks

This helps ensure the model performs well on unseen MRI data.

---

# 🔧 Configuration Parameters

| Parameter | Value |
|---|---|
| IMG_SIZE | (128,128) |
| BATCH_SIZE | 16 |
| EPOCHS | 50 |
| CONFIDENCE_THRESHOLD | 40% |
| TEST_SPLIT | 10% |
| VAL_SPLIT | 10% |

---

# 📸 Sample Output

```text
Prediction: Mild Demented
Confidence: 92.45%
MRI Validation: Passed
Heatmap Generated Successfully
```

---

# 📌 Future Improvements

- ✅ Transfer Learning (VGG16, ResNet50, EfficientNet)
- ✅ Early Stopping & Model Checkpoints
- ✅ Docker Deployment
- ✅ Cloud Deployment (AWS/GCP/Azure)
- ✅ REST API Integration
- ✅ Grad-CAM++ Visualization
- ✅ 3D MRI Volume Support
- ✅ Mobile Application Integration

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core Programming |
| TensorFlow/Keras | Deep Learning |
| Flask | Web Framework |
| OpenCV | Image Processing |
| NumPy | Numerical Operations |
| Matplotlib | Visualization |
| Scikit-learn | Evaluation Metrics |

---

# 🙏 Acknowledgements

- Kaggle Alzheimer's MRI Dataset
- TensorFlow & Keras
- OpenCV Community
- Flask Framework

---

# 👨‍💻 Developer

**Prem Selva S**  
B.Tech Artificial Intelligence and Data Science  
2026 Batch  

Project: **Alzheimer's Disease Detection System using Deep Learning**

---

# ⭐ GitHub Support

If you found this project useful:

⭐ Star the repository  
🍴 Fork the project  
🧠 Contribute to improve the system

---

# 📜 License

This project is developed for educational and research purposes.
