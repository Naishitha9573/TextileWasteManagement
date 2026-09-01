# ♻️ AI Textile Waste Intelligence Platform

<p align="center">

**An AI-powered platform for intelligent textile classification, waste categorisation, garment analysis, recycling recommendations, analytics, and sustainable textile waste management.**

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)
![EfficientNet](https://img.shields.io/badge/Model-EfficientNet--B0-orange)
![DeepFashion](https://img.shields.io/badge/Dataset-DeepFashion-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)
![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED?logo=docker)
![GitHub](https://img.shields.io/badge/Version%20Control-GitHub-black?logo=github)

</p>

---

## 📌 Table of Contents

* [Overview](#-overview)
* [Problem Statement](#-problem-statement)
* [Objectives](#-objectives)
* [Key Features](#-key-features)
* [System Architecture](#-system-architecture)
* [Project Workflow](#-project-workflow)
* [Technology Stack](#-technology-stack)
* [Project Structure](#-project-structure)
* [Dataset](#-dataset)
* [Fabric Classification](#-fabric-classification)
* [EfficientNet-B0](#-efficientnet-b0)
* [Waste Categorisation](#-waste-categorisation)
* [DeepFashion Integration](#-deepfashion-integration)
* [Recommendation Engine](#-recommendation-engine)
* [Analytics Dashboard](#-analytics-dashboard)
* [API](#-api)
* [Installation](#-installation)
* [Running the Application](#-running-the-application)
* [Docker Deployment](#-docker-deployment)
* [Model Training](#-model-training)
* [Model Evaluation](#-model-evaluation)
* [Testing](#-testing)
* [GitHub Workflow](#-github-workflow)
* [Project Milestones](#-project-milestones)
* [Limitations](#-limitations)
* [Future Enhancements](#-future-enhancements)
* [Sustainability Impact](#-sustainability-impact)
* [Contributing](#-contributing)
* [License](#-license)
* [Author](#-author)

---

# 📖 Overview

The **AI Textile Waste Intelligence Platform** is an artificial intelligence and computer vision based system designed to assist with textile and garment analysis.

The platform analyses textile images using deep learning and provides information such as:

* Fabric/material classification
* Prediction confidence
* Textile waste category
* Reuse/recycling recommendation
* Garment-level analysis
* Textile waste analytics

The primary fabric classification model uses **EfficientNet-B0**, while **DeepFashion** is incorporated as a supporting fashion/garment analysis component.

The platform also includes a lightweight rule-based waste categorisation layer, an API, analytics capabilities, testing, and Docker-based deployment.

---

# ❗ Problem Statement

The textile industry produces large amounts of waste from:

* Discarded clothing
* Manufacturing waste
* Damaged garments
* Fabric scraps
* Unsuitable or mixed materials
* Improper waste segregation

Traditional textile sorting is often manual and time-consuming.

There is a need for intelligent systems that can assist in:

1. Identifying textile materials.
2. Categorising textile waste.
3. Determining potential reuse/recycling paths.
4. Analysing textile waste data.
5. Supporting sustainable decision-making.

This project addresses these challenges using **computer vision and AI-assisted textile intelligence**.

---

# 🎯 Objectives

The main objectives of the project are:

* Develop an AI-based textile image classification system.
* Classify fabric/material types using EfficientNet-B0.
* Add a simple and explainable waste categorisation system.
* Integrate DeepFashion for garment/fashion image analysis.
* Generate recycling and reuse recommendations.
* Develop an interactive dashboard.
* Provide prediction functionality through an API.
* Perform model and application testing.
* Containerize the application using Docker.
* Demonstrate an end-to-end textile waste intelligence workflow.

---

# ✨ Key Features

## 🧵 Fabric Classification

Classifies textile images into supported fabric/material categories using a trained EfficientNet-B0 model.

## ♻️ Waste Categorisation

Maps the predicted material to a simple waste-management category such as:

* Reusable
* Recyclable
* Non-Recyclable
* Other

## 👕 DeepFashion Analysis

Uses DeepFashion as an additional fashion/garment analysis component.

## 💡 Smart Recommendations

Provides suggested actions based on the predicted material and waste category.

## 📊 Analytics Dashboard

Provides visual insights into:

* Fabric distribution
* Waste categories
* Prediction confidence
* Recyclable materials
* Reusable materials
* Non-recyclable materials

## 🔌 Prediction API

Provides model inference through a REST API.

## 🐳 Docker Deployment

The application can be packaged and executed using Docker.

## 🧪 Testing

Includes testing for:

* Model prediction
* Waste categorisation
* API functionality
* Application workflow

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │       USER          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Image Upload UI   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │ Image Preprocessing    │
                       │ Resize / Normalize     │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │    EfficientNet-B0     │
                       │ Fabric Classification  │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Predicted Fabric Type  │
                       │ + Confidence Score     │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Waste Categorisation   │
                       │ Rule-Based Mapping      │
                       └───────────┬────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
                Reusable      Recyclable    Non-Recyclable
                    │              │              │
                    └──────────────┼──────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Recommendation Engine  │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │    DeepFashion         │
                       │ Garment Analysis        │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Analytics Dashboard    │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ API / Docker Deployment│
                       └────────────────────────┘
```

---

# 🔄 Project Workflow

```text
Upload Textile Image
        │
        ▼
Image Validation
        │
        ▼
Image Preprocessing
        │
        ▼
EfficientNet-B0
        │
        ▼
Fabric Classification
        │
        ▼
Confidence Score
        │
        ▼
Waste Categorisation
        │
        ▼
Recommendation
        │
        ▼
DeepFashion Garment Analysis
        │
        ▼
Analytics
        │
        ▼
Final Result
```

---

# 🛠️ Technology Stack

| Technology      | Purpose                      |
| --------------- | ---------------------------- |
| Python          | Core development             |
| PyTorch         | Deep learning                |
| EfficientNet-B0 | Fabric classification        |
| DeepFashion     | Garment/fashion analysis     |
| OpenCV          | Image processing             |
| NumPy           | Numerical processing         |
| Pandas          | Data processing              |
| Matplotlib      | Data visualisation           |
| FastAPI         | REST API                     |
| Streamlit       | Web interface/dashboard      |
| Docker          | Containerization             |
| Git             | Version control              |
| GitHub          | Repository and collaboration |

---

# 📂 Project Structure

```text
AI-Textile-Waste-Intelligence-Platform/
│
├── app/
│   ├── main.py
│   ├── api.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── waste_categorization.py
│   └── recommendations.py
│
├── models/
│   ├── efficientnet_b0.pth
│   └── deepfashion/
│
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── preprocessing.ipynb
│   ├── training.ipynb
│   └── evaluation.ipynb
│
├── dashboard/
│   └── dashboard.py
│
├── tests/
│   ├── test_model.py
│   ├── test_api.py
│   └── test_waste_categorization.py
│
├── outputs/
│   ├── plots/
│   ├── predictions/
│   └── reports/
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

> Adjust the structure above to match your actual repository folders.

---

# 📊 Dataset

The project uses image datasets for textile and fashion analysis.

The datasets are used for:

* Training
* Validation
* Testing
* Fabric classification
* Garment analysis

Large datasets should **not normally be uploaded directly to GitHub**.

Instead, document:

* Dataset name
* Dataset source
* Dataset license
* Number of images
* Classes
* Train/validation/test split

---

# 🧵 Fabric Classification

Fabric classification is the primary AI functionality of the platform.

The model receives an image and predicts its textile/material class.

### Pipeline

```text
Image
  ↓
Resize
  ↓
Normalize
  ↓
EfficientNet-B0
  ↓
Feature Extraction
  ↓
Classification Layer
  ↓
Fabric Class
  ↓
Confidence Score
```

### Example

```text
Input:
Cotton T-shirt image

Output:
Fabric Type : Cotton
Confidence  : 94.7%
```

---

# 🧠 EfficientNet-B0

EfficientNet-B0 is used as the primary classification architecture.

The model is suitable for this project because it provides an effective balance between:

* Classification performance
* Computational efficiency
* Model size
* Training requirements
* Deployment requirements

The project uses transfer learning to adapt a pretrained EfficientNet-B0 model to textile classification.

### Model Architecture

```text
Pretrained EfficientNet-B0
            │
            ▼
    Feature Extraction
            │
            ▼
     Custom Classifier
            │
            ▼
    Textile Classification
```

---

# ♻️ Waste Categorisation

The project implements a **simple rule-based waste categorisation layer**.

This layer converts the fabric classification result into a waste-management category.

### Example Mapping

| Fabric       | Waste Category |
| ------------ | -------------- |
| Cotton       | Recyclable     |
| Polyester    | Recyclable     |
| Denim        | Reusable       |
| Silk         | Reusable       |
| Mixed Fabric | Non-Recyclable |
| Unknown      | Other          |

### Example Implementation

```python
WASTE_MAP = {
    "cotton": "Recyclable",
    "polyester": "Recyclable",
    "denim": "Reusable",
    "silk": "Reusable",
    "mixed": "Non-Recyclable"
}

waste_category = WASTE_MAP.get(
    predicted_fabric.lower(),
    "Other"
)
```

This approach is:

* Simple
* Explainable
* Easy to maintain
* Easy to integrate with the classification model

> **Important:** Actual recyclability depends on factors such as contamination, blends, construction, and available recycling processes. The project's mapping is a simplified decision-support layer.

---

# 👕 DeepFashion Integration

DeepFashion is included as a supporting fashion/garment analysis component.

The purpose is to extend the system beyond fabric-level classification and provide garment-related analysis.

### DeepFashion Workflow

```text
Garment Image
      │
      ▼
Preprocessing
      │
      ▼
DeepFashion Model
      │
      ▼
Garment/Fashion Analysis
      │
      ▼
Additional Textile Intelligence
```

DeepFashion complements the EfficientNet-B0 fabric classifier rather than replacing it.

---

# 💡 Recommendation Engine

The recommendation engine uses the prediction and waste category to suggest an appropriate next action.

### Example

```text
Fabric:
Cotton

Waste Category:
Recyclable

Recommendation:
Textile recycling / fibre recovery
```

### Example Recommendation Mapping

| Waste Category | Recommendation                  |
| -------------- | ------------------------------- |
| Reusable       | Reuse, repair, or upcycle       |
| Recyclable     | Send for textile recycling      |
| Non-Recyclable | Appropriate disposal/inspection |
| Other          | Manual inspection required      |

---

# 📊 Analytics Dashboard

The dashboard provides a high-level overview of analysed textile data.

### Dashboard Metrics

* Total images analysed
* Fabric distribution
* Waste category distribution
* Recyclable percentage
* Reusable percentage
* Non-recyclable percentage
* Prediction confidence
* Material trends

### Example

```text
+--------------------------------------------+
|       TEXTILE WASTE INTELLIGENCE           |
+----------------+---------------------------+
| Total Images   | 1,250                     |
+----------------+---------------------------+
| Recyclable     | 62%                       |
| Reusable       | 24%                       |
| Other          | 14%                       |
+----------------+---------------------------+
```

---

# 🔌 API

The platform can expose prediction functionality through FastAPI.

## Start API

```bash
uvicorn app.api:app --reload
```

The API is available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

## Example API Request

```text
POST /predict
```

The request contains a textile image.

---

## Example API Response

```json
{
    "fabric": "cotton",
    "confidence": 0.947,
    "waste_category": "Recyclable",
    "recommendation": "Send the material for textile recycling."
}
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

```bash
cd YOUR_REPOSITORY
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

If the project uses Streamlit:

```bash
streamlit run app/main.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# 🐳 Docker Deployment

Docker is used to package the application and its dependencies into a reproducible environment.

## Build Docker Image

```bash
docker build -t textile-waste-intelligence .
```

## Run Docker Container

```bash
docker run -p 8501:8501 textile-waste-intelligence
```

Open:

```text
http://localhost:8501
```

---

# 🐳 Docker Compose

If Docker Compose is configured:

```bash
docker compose up --build
```

To stop:

```bash
docker compose down
```

---

# 🧪 Model Training

The fabric classification model follows the following pipeline:

```text
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Train / Validation / Test Split
   │
   ▼
Image Augmentation
   │
   ▼
EfficientNet-B0
   │
   ▼
Transfer Learning
   │
   ▼
Model Training
   │
   ▼
Validation
   │
   ▼
Evaluation
   │
   ▼
Best Model Checkpoint
```

---

# 📈 Model Evaluation

The model should be evaluated using appropriate classification metrics.

### Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Training loss
* Validation loss
* Training accuracy
* Validation accuracy

### Results Template

```text
========================================
       MODEL EVALUATION RESULTS
========================================

Accuracy      : XX.XX%
Precision     : XX.XX%
Recall        : XX.XX%
F1 Score      : XX.XX%

========================================
```

> Replace `XX.XX%` with the actual values obtained from your model.

---

# 🔍 Confidence-Based Prediction

To avoid presenting uncertain predictions as reliable results, the system can use a confidence threshold.

Example:

```python
if confidence < 0.60:
    result = "Low confidence - manual inspection required"
else:
    result = predicted_class
```

This provides a safer approach when the model is uncertain.

---

# 🧪 Testing

Testing is performed across multiple components.

## Model Testing

Tests include:

* Model loading
* Image preprocessing
* Prediction generation
* Confidence calculation
* Output validation

## Waste Categorisation Testing

Tests include:

* Valid fabric mapping
* Unknown fabric handling
* Category generation
* Recommendation generation

## API Testing

Tests include:

* Valid image upload
* Invalid image upload
* Missing image
* Prediction response
* Error handling

## End-to-End Testing

```text
Upload Image
     ↓
Prediction
     ↓
Waste Category
     ↓
Recommendation
     ↓
Final Result
```

---

# 📋 Example End-to-End Result

```text
========================================
        TEXTILE ANALYSIS RESULT
========================================

Fabric Type:
Cotton

Confidence:
94.7%

Waste Category:
Recyclable

Recommended Action:
Textile recycling / fibre recovery

========================================
```

---

# 🌍 Real-World Applications

The platform can potentially be used in:

* Textile recycling centres
* Garment manufacturing
* Fashion companies
* Clothing collection centres
* Sustainability departments
* Waste-management organisations
* Circular fashion initiatives
* Research and educational environments

---

# 🌱 Sustainability Impact

The project supports the concept of a circular textile economy.

Potential benefits include:

* Better textile segregation
* Increased reuse opportunities
* Improved recycling identification
* Reduced unnecessary disposal
* Data-driven textile analysis
* Support for sustainable decision-making
* Increased awareness of textile waste

---

# ⚠️ Limitations

The current system has several limitations:

1. Model performance depends on dataset quality.
2. Similar fabrics may be difficult to distinguish.
3. Real-world images can differ significantly from training images.
4. Lighting, background, pose, and image quality can affect predictions.
5. Waste categorisation is currently simplified.
6. Actual recyclability cannot always be determined from fabric type alone.
7. Some textile blends require additional material analysis.
8. DeepFashion processing can require significant computational resources.
9. AI predictions should be considered decision-support information rather than professional waste-management certification.

---

# 🔮 Future Enhancements

Future versions can include:

* Automatic garment-condition detection
* Torn/damaged garment detection
* Fabric blend identification
* Multi-label material classification
* Object detection
* Textile segmentation
* Real-time camera classification
* Mobile application
* Cloud deployment
* Automated carbon-footprint estimation
* Waste quantity estimation
* Recycling-centre recommendation
* Location-based recycling centre integration
* Advanced DeepFashion integration
* Database-backed analytics
* User authentication
* Automated report generation
* Continuous model improvement

---

# 📌 Project Milestones

## Milestone 1 — Project Foundation

* [x] Project requirement analysis
* [x] Problem definition
* [x] Dataset research
* [x] Development environment
* [x] Repository setup
* [x] Initial architecture

---

## Milestone 2 — AI Model Development

* [x] Dataset preparation
* [x] Image preprocessing
* [x] EfficientNet-B0 implementation
* [x] Transfer learning
* [x] Model training
* [x] Model evaluation
* [x] Prediction pipeline

---

## Milestone 3 — Intelligence and Integration

* [x] Waste categorisation
* [ ] DeepFashion integration
* [ ] Recommendation engine
* [ ] Prediction API
* [ ] Analytics dashboard
* [ ] Testing and validation

---

## Final Deployment

* [ ] Docker configuration
* [ ] End-to-end integration
* [ ] Production testing
* [ ] Final documentation
* [ ] GitHub finalisation
* [ ] Final presentation
* [ ] Project demonstration

> Update the checkboxes based on the actual implementation before submitting the repository.

---

# 🔐 GitHub and Large Files

Large datasets and model checkpoints should generally not be committed directly to GitHub.

Example `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]

# Virtual environment
venv/
.env/

# Datasets
data/
datasets/

# Model checkpoints
*.pth
*.pt
*.onnx

# Logs
*.log

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

For large model files, consider using an appropriate model/artifact storage solution.

---

# 🔄 GitHub Development Workflow

## Check Repository Status

```bash
git status
```

## Add Changes

```bash
git add .
```

## Commit Changes

```bash
git commit -m "Add textile waste categorisation"
```

## Push Changes

```bash
git push origin main
```

---

# 📝 Recommended Commit History

A clean commit history can look like:

```text
Initial project structure

Add textile dataset preprocessing

Implement EfficientNet-B0 classifier

Add model training pipeline

Add model evaluation

Add prediction pipeline

Add waste categorisation

Add recommendation engine

Integrate DeepFashion

Add prediction API

Add analytics dashboard

Add application tests

Add Docker deployment

Update documentation
```

---

# 🤝 Contributing

Contributions are welcome.

### 1. Fork the repository

### 2. Create a feature branch

```bash
git checkout -b feature/new-feature
```

### 3. Make your changes

### 4. Commit

```bash
git commit -m "Add new feature"
```

### 5. Push

```bash
git push origin feature/new-feature
```

### 6. Open a Pull Request

---

# 📜 License

This project is intended primarily for educational, research, and demonstration purposes.

Dataset licenses and third-party model licenses remain subject to their respective owners and terms.

---

# 👩‍💻 Author

## Naishitha Kandukuri

**AI Textile Waste Intelligence Platform**

---

# ⭐ Project Summary

The **AI Textile Waste Intelligence Platform** combines computer vision, deep learning, rule-based waste categorisation, fashion-image analysis, recommendations, analytics, APIs, and containerized deployment.

```text
             AI TEXTILE WASTE
                 INTELLIGENCE
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
   Fabric AI     DeepFashion     Analytics
       │              │              │
       └──────────────┼──────────────┘
                      ▼
             Waste Categorisation
                      │
                      ▼
              Recommendations
                      │
                      ▼
                   API
                      │
                      ▼
                  Docker
```

The ultimate goal is to demonstrate how AI can support **textile identification, waste segregation, reuse, recycling, and sustainable decision-making**.

---

# 🚀 Vision

> **Transform textile waste management from manual identification into an AI-assisted, data-driven and sustainable workflow.**

---

<p align="center">

### ♻️ AI for Sustainable Textiles

**Built with Python • PyTorch • EfficientNet-B0 • DeepFashion • FastAPI • Streamlit • Docker**

</p>
