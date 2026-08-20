# Diabetes Risk Prediction & Bangladeshi Diet Recommendation System

An end-to-end machine learning pipeline for ternary diabetes risk classification (Non-Diabetic / Pre-Diabetic / Diabetic), coupled with an explainable AI layer and a Retrieval-Augmented Generation (RAG) based personalized diet recommendation system grounded in the official Food Composition Table for Bangladesh (FCTB).

## Overview

This project addresses two gaps in existing diabetes-prediction and diet-recommendation research:
- Most prediction pipelines use binary classification and apply preprocessing before the train/test split, causing data leakage.
- Existing RAG/LLM-based dietary systems are not conditioned on a clinical risk tier or grounded in region-specific food data.

This system combines a **leakage-free, ternary Stacking Ensemble classifier** with a **knowledge-graph-grounded RAG diet generation pipeline**, deployed end-to-end in a single Streamlit web application.

## Key Features

- **Ternary diabetes risk prediction** (Non-Diabetic / Pre-Diabetic / Diabetic) using a Stacking Ensemble of Gradient Boosting, XGBoost, and LightGBM (97.09% accuracy, 92.22% macro F1)
- **Leakage-free pipeline**: train/test split performed before SMOTE, scaling, and feature selection
- **Dual explainability**: global SHAP + local LIME explanations
- **DiCE counterfactual guidance**: actionable, clinically modifiable "what-if" suggestions (e.g., target HbA1c to shift risk tier)
- **Knowledge-graph-grounded RAG diet generation**: an rdflib/SPARQL knowledge graph over the official Food Composition Table for Bangladesh (FCTB), constraining a Gemini LLM to generate hallucination-resistant, personalized diet charts
- **Quantitative faithfulness evaluation**: food-grounding rate, portion fidelity, and LLM-judge (RAGAS-inspired) faithfulness scoring
- **Single Streamlit web application** covering the full pipeline from clinical input to personalized diet chart

## Datasets

1. **Diabetes Clinical Dataset (100K rows)** — [Kaggle](https://www.kaggle.com/datasets/ziya07/diabetes-clinical-dataset100k-rows) — used for diabetes risk prediction.
2. **Food Composition Table for Bangladesh (FCTB)** — Institute of Nutrition and Food Science (INFS), University of Dhaka, via [FAO/INFOODS](https://www.fao.org/infoods/infoods/tables-and-databases/asia/en/) — used for diet recommendation.

## Tech Stack

- **Language**: Python 3.14
- **ML/Explainability**: scikit-learn, XGBoost, LightGBM, SHAP, LIME, DiCE-ML, Optuna
- **Knowledge Graph**: rdflib, SPARQL
- **LLM**: Google Gemini API
- **Web App**: Streamlit

## Project Structure
