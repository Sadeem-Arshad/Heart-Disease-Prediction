#  Heart Disease Risk Prediction Web App

A **machine learning–based web application** that predicts the risk of heart disease using the **CDC BRFSS 2015 dataset**.
The model achieves **81.7% recall**, successfully identifying **more than 4 out of 5 patients** who have heart disease.

This project was developed as part of the **Introduction to Data Science (DL2001)** course at **FAST-NUCES Lahore**.

---

#  Project Overview

Heart disease is one of the leading causes of death worldwide. Early risk assessment can help individuals take preventive measures before serious complications occur.

This project uses **machine learning** to analyze health and lifestyle indicators and predict whether a person is at risk of heart disease.

The system is deployed as a **web-based interactive dashboard** built with **Streamlit**, allowing users to enter their health information and instantly receive a prediction.

---

#  Key Features

* Interactive **web-based prediction interface**
* **Random Forest model** trained on health survey data
* **81.7% recall**, effectively detecting most heart disease cases
* **Real-time risk assessment**
* **Comprehensive feature analysis**
* **User-friendly Streamlit dashboard**

---

#  Technologies Used

* **Python 3.8+**
* **Streamlit** – Web application interface
* **scikit-learn** – Machine learning models
* **pandas** – Data processing and manipulation
* **NumPy** – Numerical computations
* **pickle** – Model serialization and deployment

---

#  Dataset Information

**Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015

* **Total Samples:** 253,680 health survey responses
* **Features:** 22 health and demographic indicators
* **Target Variable:** Heart Disease or Attack (Binary Classification)
* **Class Imbalance Handling:** Undersampling to create a **50–50 balanced dataset**

The dataset includes indicators such as:

* BMI
* Smoking habits
* Physical activity
* Diabetes
* Alcohol consumption
* General health condition
* Age and demographic information

---

#  Machine Learning Model

The model used in this project is a **Random Forest Classifier**, chosen for its robustness and ability to handle complex relationships between features.

**Performance Metric:**

* **Recall:** 81.7%
  This means the model successfully identifies **over 4 out of 5 patients** who actually have heart disease.

---

#  Contributors

Developed by:

* **Sadeem Arshad** (24L-2502)
* **Hamza Sheikh** (24L-2500)
* **Abuzar Rizwan** (24L-2535)

FAST National University of Computer and Emerging Sciences (FAST-NUCES), Lahore.
