# 🌱 Sugarcane Leaf Disease Detection Using Hybrid RGB + LBP and MobileNetV3

An AI-based web application for detecting sugarcane leaf diseases from uploaded leaf images. The system combines deep learning-based visual features with Local Binary Pattern (LBP) texture information and provides predictions through an easy-to-use Streamlit web interface.

---

## 📌 Project Overview

Sugarcane is an important agricultural crop, but several diseases can significantly reduce crop yield and quality. Early and accurate disease identification can help farmers take suitable preventive and treatment measures.

This project develops an image-based Artificial Intelligence system that analyzes sugarcane leaf images and predicts the corresponding disease category.

The project includes:

- Sugarcane leaf image preprocessing
- Leaf-image validation
- Deep learning using **MobileNetV3Small**
- RGB feature extraction
- **Local Binary Pattern (LBP)** texture feature extraction
- Hybrid feature integration
- Model training and evaluation
- Ablation study
- Confusion matrix and ROC analysis
- Streamlit-based web deployment
- Image upload and disease prediction interface

---

## 🎯 Objectives

The main objectives of this project are:

1. To develop an AI-based system for sugarcane leaf disease detection.
2. To classify sugarcane leaf images into different disease and leaf-condition categories.
3. To use transfer learning with MobileNetV3Small for efficient image classification.
4. To incorporate LBP-based texture information along with RGB image information.
5. To evaluate the model using accuracy, macro F1-score, AUC, confusion matrix and ROC curves.
6. To provide a simple web interface where users can upload an image and receive a prediction.
7. To deploy the trained model as a web application using Streamlit.

---

## 🧠 Methodology

The overall workflow of the project is:

```text
Sugarcane Leaf Image
        ↓
Image Upload
        ↓
Leaf Image Validation
        ↓
Image Preprocessing
        ↓
RGB Feature Extraction
        +
LBP Texture Feature Extraction
        ↓
Hybrid Feature Representation
        ↓
MobileNetV3Small-Based Model
        ↓
Classification
        ↓
Predicted Disease / Leaf Condition
        ↓
Streamlit Web Interface
```

---

## 📂 Dataset

The project uses a Sugarcane Leaf Image Dataset containing disease and leaf-condition categories.

The dataset contains the following categories:

### Disease Classes

1. Banded Chlorosis
2. Brown Spot
3. BrownRust
4. Grassy shoot
5. Pokkah Boeng
6. Sett Rot
7. smut
8. Viral Disease
9. Yellow Leaf

### Additional Leaf Conditions

10. Healthy Leaves
11. Dried Leaves

Therefore, the implementation can be treated as an **11-class image classification problem** when both healthy and dried leaves are included.

> **Note:** If your final training dataset contains a different number of classes, update this section and the class names according to the final dataset used for training.

---

## 🗂️ Suggested Dataset Structure

```text
dataset/
│
├── Diseases/
│   ├── Banded Chlorosis/
│   ├── Brown Spot/
│   ├── BrownRust/
│   ├── Grassy shoot/
│   ├── Pokkah Boeng/
│   ├── Sett Rot/
│   ├── smut/
│   ├── Viral Disease/
│   └── Yellow Leaf/
│
├── Healthy Leaves/
│
└── Dried Leaves/
```

---

## 🔄 Data Splitting

The dataset was divided into:

- **70% Training**
- **15% Validation**
- **15% Testing**

The training data is used to learn model parameters, validation data is used for monitoring model performance during training, and the test data is used for final evaluation.

---

## 🖼️ Image Preprocessing

Uploaded images are converted into RGB format and resized to:

```text
224 × 224 pixels
```

MobileNetV3 preprocessing is applied before passing the image to the deep learning model.

A simplified preprocessing pipeline is:

```text
Input Image
    ↓
RGB Conversion
    ↓
Resize to 224 × 224
    ↓
MobileNetV3 Preprocessing
    ↓
Model Input
```

---

## 🍃 Leaf Validation

Before disease classification, the application performs a basic leaf validation step.

The uploaded image is converted to HSV color space and a green-region mask is generated.

The implementation uses approximately:

```python
lower_green = np.array([25, 40, 40])
upper_green = np.array([95, 255, 255])
```

A green-pixel ratio is calculated to determine whether the uploaded image is likely to contain a leaf.

This helps reduce incorrect predictions from completely unrelated images.

> This validation is a lightweight image-content check and should not be considered a guaranteed botanical leaf detector.

---

# 🧩 Hybrid RGB + LBP Feature Extraction

A major part of the project is the combination of RGB image information and texture information.

## RGB Features

RGB information captures visual characteristics such as:

- Color
- Disease-related discoloration
- Leaf appearance
- Spots and affected regions

## LBP Features

**Local Binary Pattern (LBP)** is used to capture local texture patterns.

The implementation uses:

```text
P = 8
R = 1
Method = uniform
```

LBP is useful for representing texture differences that may appear in diseased regions of leaves.

## Hybrid Representation

The project combines deep visual features with texture information to improve disease classification.

```text
RGB Image
   ↓
MobileNetV3Small
   ↓
Deep Visual Features
        +
LBP Texture Features
   ↓
Feature Fusion
   ↓
Classification Layer
   ↓
Disease Prediction
```

---

# 🤖 Model Architecture

The main deep learning backbone used in this project is:

## MobileNetV3Small

MobileNetV3Small is a lightweight convolutional neural network suitable for image classification and resource-constrained environments.

The model uses transfer learning so that previously learned visual representations can be adapted to the sugarcane leaf dataset.

Major components include:

```text
Input Image
     ↓
MobileNetV3Small
     ↓
Feature Extraction
     ↓
Global Average Pooling
     ↓
Feature Processing / Attention
     ↓
Hybrid Feature Integration
     ↓
Dense Classification Layer
     ↓
Softmax Output
```

Depending on the final implementation, the architecture also includes feature-processing operations such as:

- GlobalAveragePooling2D
- Reshape
- Multiply
- Conv2D
- Concatenate
- Add

---

# ⚙️ Training

The model is trained using TensorFlow/Keras.

Training includes:

- Transfer learning
- Image preprocessing
- Early stopping
- Up to 50 training epochs
- Validation monitoring
- Model checkpointing

Early stopping is used to reduce unnecessary training and help prevent overfitting.

---

# 🔬 Ablation Study

An ablation study was performed to understand the contribution of different components of the proposed system.

| Configuration | Accuracy | Macro F1 | Macro AUC |
|---|---:|---:|---:|
| A1 – Full Model | 88.2% | 88.2% | 0.994 |
| A2 | 88.2% | 88.1% | 0.993 |
| A4 | 87.2% | 87.1% | 0.989 |
| A3 | 86.4% | 86.3% | 0.988 |
| A5 | 86.2% | 86.1% | 0.989 |

The full configuration achieved the strongest overall performance among the evaluated configurations.

> If your final experiment produces different metrics, replace the values above with the final reported results.

---

# 📊 Evaluation Metrics

The model is evaluated using multiple performance metrics.

### Accuracy

Measures the proportion of correctly classified images.

### Macro F1-Score

Calculates the F1-score for each class and gives equal importance to every class.

This is particularly useful when the dataset is imbalanced.

### Macro AUC

Measures the overall ability of the classifier to distinguish between classes using the ROC-based area under the curve.

### Confusion Matrix

A confusion matrix shows correct and incorrect predictions for each class.

### ROC Curve

ROC curves are used to analyze the classification performance across different decision thresholds.

---

# 📁 Project Structure

A recommended GitHub repository structure is:

```text
sugarcane-leaf-disease-detection/
│
├── app.py
├── requirements.txt
├── README.md
├── runtime.txt
├── .gitignore
│
├── model/
│   └── best_model.keras
│
├── utils/
│   └── util.py
│
├── assets/
│   ├── logo.png
│   └── sample_images/
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── metrics.json
│   └── Final_Summary.csv
│
└── notebooks/
    └── model_training.ipynb
```

> The exact structure can be different depending on your implementation. Make sure every file referenced by `app.py` is included in the GitHub repository or otherwise made available to the deployment environment.

---

# 💻 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep learning |
| Keras | Model development |
| MobileNetV3Small | Transfer learning backbone |
| NumPy | Numerical computation |
| Pandas | Data processing |
| OpenCV | Image processing |
| Pillow | Image loading and conversion |
| scikit-image | LBP feature extraction |
| scikit-learn | Evaluation metrics |
| Matplotlib | Visualization |
| Streamlit | Web application and deployment |
| GitHub | Source-code hosting and version control |

---

# 📦 Installation

## 1. Clone the Repository

Open Command Prompt or Terminal:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

Move into the project directory:

```bash
cd YOUR-REPOSITORY
```

---

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

A typical `requirements.txt` for this project can contain:

```text
streamlit
tensorflow
keras
numpy
pandas
pillow
opencv-python-headless
scikit-learn
scikit-image
matplotlib
```

If your final application requires specific package versions, use the tested versions from your working environment instead.

---

# ▶️ Run the Application Locally

From the project root directory:

```bash
streamlit run app.py
```

If the `streamlit` command is not recognized, use:

```bash
python -m streamlit run app.py
```

The application will normally open in your browser at:

```text
http://localhost:8501
```

---

# 🌐 Web Application

The deployed Streamlit application allows users to:

1. Open the web application.
2. Upload a sugarcane leaf image.
3. Validate the uploaded image.
4. Preprocess the image.
5. Pass the image through the trained model.
6. Generate the predicted class.
7. Display the prediction to the user.

### Application Workflow

```text
User
 ↓
Open Web Application
 ↓
Upload Sugarcane Leaf Image
 ↓
Image Validation
 ↓
Preprocessing
 ↓
MobileNetV3 + Hybrid Features
 ↓
Prediction
 ↓
Disease / Leaf Condition Result
```

---

# ☁️ Streamlit Cloud Deployment

The application can be deployed using Streamlit Community Cloud.

## Step 1 – Upload the Project to GitHub

Create a new GitHub repository.

For example:

```text
sugarcane-leaf-disease-detection
```

Upload:

```text
app.py
requirements.txt
runtime.txt
model/
utils/
assets/
README.md
```

Do not upload unnecessary files such as:

```text
venv/
__pycache__/
.ipynb_checkpoints/
```

---

## Step 2 – Add `requirements.txt`

Create:

```text
requirements.txt
```

and add the Python packages required by the application.

Example:

```text
streamlit
tensorflow
keras
numpy
pandas
pillow
opencv-python-headless
scikit-learn
scikit-image
matplotlib
```

---

## Step 3 – Add `runtime.txt`

If the application is tested with Python 3.11, create:

```text
runtime.txt
```

with:

```text
python-3.11
```

Use the Python version that is compatible with your final TensorFlow/Keras environment.

---

## Step 4 – Check Model Paths

If the model is located at:

```text
model/best_model.keras
```

the Python code should reference:

```python
model_path = "model/best_model.keras"
```

If using an H5 model:

```python
model_path = "model/best_model.h5"
```

The capitalization and folder names must exactly match the files in GitHub.

---

## Step 5 – Deploy

Open Streamlit Community Cloud and connect your GitHub repository.

Select:

```text
Repository → Branch → app.py
```

Then click:

```text
Deploy
```

Streamlit will install the packages from `requirements.txt` and start the application.

---

# 🔗 Live Application

**Streamlit Web App:**

```text
https://drwsddynszwyaiqjg6bhob.streamlit.app/
```

Example format:

```text
https://your-app-name.streamlit.app
```

Replace the placeholder with your actual deployed application URL.

---

# 🛠️ Troubleshooting Streamlit Deployment

## Error Running App

If Streamlit displays:

```text
Oh no.

Error running app.
```

the browser page is showing only a generic error.

Open the application management page and check the application logs.

The logs normally contain the actual Python exception, such as:

```text
ModuleNotFoundError
FileNotFoundError
ImportError
MemoryError
```

---

## `ModuleNotFoundError`

Example:

```text
ModuleNotFoundError: No module named 'cv2'
```

Add the required package to:

```text
requirements.txt
```

For OpenCV on Streamlit Cloud, use:

```text
opencv-python-headless
```

Then redeploy the application.

---

## `FileNotFoundError`

Example:

```text
FileNotFoundError: best_model.keras
```

Check that:

1. The model exists in GitHub.
2. The model path in `app.py` is correct.
3. The folder name is correct.
4. Uppercase/lowercase spelling matches exactly.

---

## TensorFlow/Keras Compatibility

If the model fails to load after deployment, verify that the TensorFlow/Keras versions used for training and deployment are compatible.

It is recommended to test the application locally using the same dependency versions specified in `requirements.txt`.

---

## Large Model Files

Large model files may cause GitHub or deployment limitations.

If the trained model is too large for normal GitHub storage, consider an appropriate large-file or external model-storage solution and load the model securely at runtime.

Do not commit passwords, API keys, access tokens or other secrets to GitHub.

---

# 🔐 Security

Never upload sensitive information such as:

```text
API keys
Passwords
Access tokens
Private credentials
Secret configuration files
```

Use Streamlit secrets or the deployment platform's secret-management facility when credentials are required.

---

# 📸 Example Usage

```text
1. Open the deployed application
2. Select "Upload Image"
3. Choose a sugarcane leaf image
4. Wait for image processing
5. View the predicted class
```

The system is intended to provide an AI-assisted prediction based on the uploaded image.

---

# 📈 Results

The proposed system demonstrated strong classification performance on the evaluated sugarcane leaf dataset.

The reported full-model ablation configuration achieved:

```text
Accuracy   : 88.2%
Macro F1   : 88.2%
Macro AUC  : 0.994
```

The exact performance may vary depending on the dataset split, preprocessing, training configuration and model version.

---

# 🌾 Practical Benefits

The proposed application can help:

- Farmers perform preliminary disease identification.
- Reduce the time required for manual visual inspection.
- Provide an accessible image-based prediction interface.
- Support early identification of potentially diseased leaves.
- Demonstrate the practical use of AI in agriculture.

The prediction should be considered an AI-assisted indication and not a replacement for expert agricultural diagnosis.

---

# 🚀 Future Scope

Future improvements can include:

- Increasing the size and diversity of the dataset.
- Collecting images under real field conditions.
- Improving performance on visually similar diseases.
- Adding disease severity estimation.
- Adding treatment and prevention recommendations.
- Developing a mobile Android application.
- Supporting multiple regional languages.
- Adding farmer-friendly voice interaction.
- Integrating weather and environmental information.
- Improving leaf/background segmentation.
- Using federated learning for privacy-preserving distributed training.
- Optimizing the model for edge devices and mobile deployment.
- Adding explainable AI techniques such as Grad-CAM.
- Providing location-based agricultural assistance.

---

# 📚 Research Contribution

The project focuses on combining:

```text
Deep Learning
      +
Transfer Learning
      +
RGB Visual Information
      +
LBP Texture Information
      +
Web Deployment
```

The hybrid approach is designed to use both visual and local texture characteristics of sugarcane leaves for disease classification.

---

# 👨‍💻 Author

**Mritunjoy Paul**

B.Tech – Computer Science and Engineering (Artificial Intelligence & Machine Learning)

---

# 📄 License

This project is intended for academic, educational and research purposes.

If you plan to release the project publicly, add an appropriate open-source license such as MIT License after confirming that all included datasets, images and third-party components permit the intended use.

---

# ⭐ Acknowledgements

This project uses open-source technologies and machine learning libraries including TensorFlow, Keras, Streamlit, NumPy, OpenCV, Pillow, scikit-image and scikit-learn.

---

# 📌 Citation

If this project is used in academic work, please cite the associated research paper/project report.

```text
Mritunjoy Paul,
"AI-Based Sugarcane Leaf Disease Detection Using Hybrid RGB + LBP Features and MobileNetV3",
Academic Project / Research Work.
```

---

## ⭐ If You Find This Project Useful

Consider giving the repository a ⭐ on GitHub and sharing it with others working in AI, machine learning and smart agriculture.
