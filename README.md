# Electricity Theft Detection System

## Purpose

Electricity theft is a major issue faced by electricity distribution companies, leading to significant financial losses, power instability, and unfair billing for honest consumers.  
This project provides an automated solution for detecting electricity theft using machine learning techniques.  
By analyzing electricity consumption behavior such as usage patterns, voltage fluctuations, bill payment delay, and unusual usage spikes, the system predicts the likelihood of electricity theft and assists authorities in effective decision-making.

---

## Model Overview

The machine learning model used in this project is a **Random Forest Classifier**.  
This model was chosen due to its robustness, high accuracy, and ability to handle complex datasets with multiple input features.

The model is trained primarily on **electricity consumption behavior**, making it suitable for real-world deployment scenarios.

---

### Key Techniques Used:

1. **SMOTE (Synthetic Minority Over-sampling Technique)**  
   SMOTE is used to handle class imbalance in the dataset by generating synthetic samples for the minority class (theft cases).  
   This improves the model’s ability to correctly identify electricity theft.

2. **Grid Search for Hyperparameter Tuning**  
   The model is optimized using `GridSearchCV`, which searches for the best combination of hyperparameters to improve performance.

   **Tuned Hyperparameters include:**  
   - `n_estimators`: Number of trees in the forest  
   - `max_depth`: Maximum depth of each tree  
   - `min_samples_split`: Minimum samples required to split a node  
   - `min_samples_leaf`: Minimum samples required at a leaf node  

---

### Model Pipeline:

The complete machine learning pipeline includes the following steps:

- Data preprocessing and feature scaling  
- Handling class imbalance using SMOTE  
- Training the Random Forest Classifier  
- Model evaluation and optimization  
- Saving the trained pipeline using Pickle  

This ensures consistent preprocessing and prediction during deployment.

---

## Web Application

The trained machine learning model is deployed using **Streamlit**, providing an interactive and user-friendly web interface.

The application allows users to:
- Enter electricity consumption details  
- Obtain theft or no-theft prediction  
- View theft risk percentage  
- Understand key factors influencing the prediction  

---

## How to Run the Streamlit App

1. **Install Dependencies**

   Ensure all required dependencies are installed by running:
   ```bash
   pip install -r requirements.txt


## How to Run the Streamlit App

1. **Install Dependencies**:
   Before running the Streamlit app, make sure to install all necessary dependencies. You can do this by running the following command:
   ```bash
   pip install -r requirements.txt

2. **Run the Streamlit App**:
Once the dependencies are installed, you can run the Streamlit app using the following command:
```bash
streamlit run app.py
```

This will launch the app in your web browser, typically at http://localhost:8501.
