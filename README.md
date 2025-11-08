# Multiclass Fish Image Classification

## Domain
Image Classification

---

## Problem Statement
This project focuses on classifying fish images into multiple species using deep learning. The task includes:
- Training a CNN from scratch,
- Leveraging transfer learning with pre-trained models,
- Fine-tuning the best model,
- Saving trained models for later use,
- Deploying a Streamlit application for real-time predictions from user-uploaded images.

---

## Business Use Cases
- **Enhanced Accuracy:** Find the best model architecture for fish classification.
- **Deployment Ready:** Provide a web app for real-time predictions (useful for fisheries, retailers, educational tools).
- **Model Comparison:** Compare multiple models and select the most suitable one.

---

## Project Approach
1. **Data Preprocessing and Augmentation**
   - Rescale images to [0, 1].
   - Apply augmentations: rotation, zoom, width/height shift, flip, brightness adjustments.

2. **Model Training**
   - Train a CNN from scratch.
   - Train five pre-trained models: VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0.
   - Use transfer learning and fine-tune the top layers.

3. **Model Saving**
   - Save best versions of models as `.h5`.

4. **Model Evaluation**
   - Compare models using: Accuracy, Precision, Recall, F1-score, Confusion Matrix.
   - Visualize training history (accuracy & loss).

5. **Deployment**
   - Build Streamlit app to upload image, show predicted class & confidence, and display probabilities.

---

## Dataset
- Images organized in folders per class (train / val / test).

## Workflow

### Step 1: Data Loading
- Images are organized into train and validation folders by class.  
- Augmentation and preprocessing applied via `ImageDataGenerator`.

### Step 2: Model Training and Comparison
- Trains CNN and 5 transfer learning models.  
- Evaluates using accuracy and F1-score.  

### Step 3: Fine-Tuning the Best Model
- MobileNetV2 selected as the best model.  
- Unfreezes top layers and retrains for 5 epochs (CPU) or 15 epochs (GPU).

### Step 4: Streamlit Deployment
- Upload an image for prediction.  
- Displays the predicted class name and confidence score.

---
## Download Dataset and Trained Model

To keep this repository lightweight, the dataset and trained model are hosted externally on Google Drive.

🔗 **[Open Google Drive Folder](https://drive.google.com/drive/folders/1-c9RF9gL8-NsMmAinn2OsM_AMNaB2ScR?usp=sharing)**

The Drive folder contains:
- `data/` -> with `train`, `val`, and `test` subfolders  
- best model file -> `final_best_model_finetuned.h5`  
   

### Setup Instructions
1. Download the Drive folder or individual files.
2. Place them inside your project root so the structure looks like:

## Requirements
- tensorflow
- numpy
- matplotlib
- seaborn
- pandas
- scikit-learn
- streamlit

## Manju R
## AIML-S-WE-T-B16
