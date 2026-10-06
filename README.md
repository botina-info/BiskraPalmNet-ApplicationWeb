# BiskraPalmNet-Web
## Explainable Web-Based Intelligent Decision Support System for Date Palm Disease Monitoring

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%2FKeras-orange.svg)](https://www.tensorflow.org/)
[![XAI](https://img.shields.io/badge/XAI-Grad--CAM-purple.svg)](https://arxiv.org/abs/1610.02391)

---

## 1. Overview

**BiskraPalmNet-Web** is a web-based, explainable Intelligent Decision Support System (DSS) designed for the monitoring of date palm leaf diseases.

The system integrates a trained deep learning model, an interactive web interface, Explainable Artificial Intelligence (XAI), and agricultural decision-support functionalities into a unified workflow.

Rather than using the BiskraPalmNet model as a standalone image-classification tool, the proposed application transforms model predictions into an operational decision-support workflow:

**Image Acquisition → Image Preprocessing → AI Prediction → Confidence Estimation → Grad-CAM Explainability → Decision Support → Agricultural Recommendation**

The system is intended to support practical date palm disease monitoring, particularly in agricultural environments where rapid and accessible preliminary assessment can assist farmers and agricultural practitioners.

---

## 2. Research Context

Date palm (*Phoenix dactylifera* L.) is an important agricultural crop in arid and Saharan regions, including the Biskra region of Algeria.

Disease monitoring traditionally relies heavily on visual inspection, which may be time-consuming and dependent on the experience of the observer. Artificial intelligence and computer vision provide opportunities for supporting faster and more consistent preliminary disease recognition.

The BiskraPalmNet-Web system addresses this application perspective by connecting artificial intelligence-based image classification with explainability and agricultural decision support.

The objective is therefore not limited to predicting a disease class, but to provide an accessible interface through which the prediction can be interpreted and transformed into an appropriate monitoring or management recommendation.

---

## 3. Main Objectives

The main objectives of BiskraPalmNet-Web are to:

- provide a user-friendly web interface for date palm leaf image analysis;
- integrate the BiskraPalmNet deep learning model into an operational application;
- classify uploaded date palm leaf images into the considered health/disease categories;
- provide the model prediction together with its confidence;
- improve interpretability using Grad-CAM-based visual explanations;
- highlight image regions that contribute to the model prediction;
- connect AI predictions with decision-support recommendations;
- provide an accessible tool for preliminary date palm disease monitoring;
- demonstrate the transition from an experimental AI model to a real-world agricultural application.

---

## 4. Considered Date Palm Leaf Conditions

The application considers three classes:

1. **White Cochineal**
2. **Brown Leaf Spot**
3. **Healthy Date Palm Leaf**

The class-label mapping used by the application is defined in:

```text
class_names.txt
---

## 5.Author

Bouthina Rachachi
PhD Researcher
University Mohamed Khider, Biskra, Algeria
Research interests include:
- Artificial Intelligence in Agriculture
- Deep Learning
- Computer Vision
- Explainable Artificial Intelligence
- Smart Agriculture
- Agricultural Decision Support Systems
- Date Palm Disease Monitoring

---
