# 🍎 Fruit Quality Detection System

This project implements a **Computer Vision–based Fruit Quality Detection System** that uses **Deep Learning** to automatically identify whether an uploaded image of a fruit (Apple or Banana) is **Fresh** or **Rotten**.

Developed as part of the *Digital Image Processing* coursework, the project demonstrates how digital image preprocessing and Convolutional Neural Networks (CNNs) can be used for visual classification tasks.

---

## 🚀 Features

- Detects fruit type: **Apple** or **Banana**
- Classifies fruit quality as **Fresh** or **Rotten**
- Uses a trained **CNN model** built with TensorFlow and Keras
- Provides a simple and interactive **Gradio-based interface** for image upload and prediction
- Demonstrates core **Digital Image Processing** techniques such as resizing, normalization, and feature extraction

---

## 🧠 Tech Stack

- **Language:** Python 3.x
- **Libraries:** TensorFlow, Keras, NumPy, Matplotlib, Pillow, Gradio
- **Environment:** Google Colab / Jupyter Notebook

---

## 📂 Folder Structure

```
Fruit-Quality-Detection-System/
│
├── Fruit_Quality_Monitoring.ipynb    # Main Jupyter Notebook
├── fruit_classifier.h5               # Model for fruit type detection (linked via Google Drive)
├── apple_quality_classifier.h5       # Model for Apple freshness classification (linked via Google Drive)
├── banana_quality_classifier.h5      # Model for Banana freshness classification (linked via Google Drive)
├── dataset/                          # Dataset folder (if added locally)
│   ├── apple/
│   │   ├── fresh/
│   │   └── rotten/
│   ├── banana/
│       ├── fresh/
│       └── rotten/
└── Test/
    ├── freshapples/
    ├── rottenapples/
    ├── freshbanana/
    └── rottenbanana/
```

---

## 📦 Dataset

The dataset used for this project is available on Kaggle:  
👉 [Fruits Fresh and Rotten for Classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification?resource=download)

It contains images of various fruits, categorized into fresh and rotten classes.  
For this project, only **Apple** and **Banana** classes are used.

---

## 🔗 Model Files

Due to GitHub’s file size limit (25 MB), trained model files are hosted externally.  
Please download them using the following Google Drive links and place them in your project root folder:

- [fruit_classifier.h5](https://drive.google.com/file/d/1jQd_ta74YLn7EDHQ82tt5EhWwFYLa4zT/view?usp=sharing)
- [apple_quality_classifier.h5](https://drive.google.com/file/d/1z56tVm_9AIy7672ofTTJRYovPqGLtLnq/view?usp=sharing)
- [banana_quality_classifier.h5](https://drive.google.com/file/d/<BANANA_MODEL_ID>/view)

Or automatically download them in the notebook using:

```python
import gdown
gdown.download("https://drive.google.com/file/d/1jQd_ta74YLn7EDHQ82tt5EhWwFYLa4zT/view?usp=sharing", "fruit_classifier.h5", quiet=False)
gdown.download("https://drive.google.com/file/d/1z56tVm_9AIy7672ofTTJRYovPqGLtLnq/view?usp=sharing", "apple_quality_classifier.h5", quiet=False)
gdown.download("https://drive.google.com/file/d/11XTLPyayRPiKz6UmQnOtSJQ72rJXfCJe/view?usp=sharing", "banana_quality_classifier.h5", quiet=False)
```

---

## ⚙️ How to Run

1. **Clone this repository**
   ```bash
   git clone https://github.com/Paarth16/Fruit-Quality-Detection.git
   cd Fruit-Quality-Detection-System
   ```

2. **Install required libraries**
   ```bash
   pip install tensorflow keras numpy pillow matplotlib gradio gdown
   ```

3. **Open the notebook**
   - Run `Fruit_Quality_Monitoring.ipynb` in **Google Colab** or **Jupyter Notebook**.

4. **Launch the interface**
   Once models are downloaded or trained, execute the Gradio interface cell to open the app:
   ```python
   iface.launch()
   ```

5. **Upload an image**
   Upload a fruit image (Apple or Banana) to classify whether it is **Fresh** or **Rotten**.

---

## 🧪 Model Overview

Three CNN models are trained for this project:
1. `fruit_classifier.h5` – identifies fruit type (Apple or Banana)
2. `apple_quality_classifier.h5` – classifies Apple as Fresh or Rotten
3. `banana_quality_classifier.h5` – classifies Banana as Fresh or Rotten

Each model uses convolutional layers for feature extraction, pooling for dimensionality reduction, and dense layers for classification.

---

## 📈 Results

- **Fruit Classification Accuracy:** ~90–95%
- **Quality Classification Accuracy:** ~90%+
- Visualizations include training/validation accuracy and loss graphs.
- The system demonstrates robust performance on test images.

---

## 🧩 Digital Image Processing Concepts Used

- Image Acquisition (via Gradio upload)
- Image Preprocessing (resizing, normalization)
- Feature Extraction (through CNN filters)
- Edge and texture detection (automatically learned)
- Classification and result display

---

## 🎓 Author

**Paarth Sharma**  
Department of Computing Technolodies
SRM Institute of Science and Technology  
Subject: *Digital Image Processing (DIP)*

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
