# Flight Price Prediction

An application for predicting flight prices using machine learning. The structured branch contains the data pipeline, model, and API setup necessary to support predictions.

---

## 🚀 Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Repo Structure](#repo-structure)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Setup and Installation](#setup-and-installation)  
  - [Running the API](#running-the-api)  
- [Usage](#usage)  
- [Testing](#testing)  
- [Tech Stack](#tech-stack)  
- [License](#license)  

---

## Overview

This project uses historical flight data and statistical / machine learning methods to predict flight ticket prices. The prediction model is exposed via a REST API so that it can be used by frontends or other services.  

---

## Features

- Data pipeline for cleaning, feature engineering, model training.  
- Predictive model serialized (`.pkl`) for inference.  
- REST API endpoints to get price predictions.  
- Logging configuration for debugging / monitoring.  
- Structured layout, tests, and notebooks for experimentation.  

---

## Repo Structure

Here’s a quick rundown of the folders & files:

├── data/ # Raw or processed data used for training/experimentation
├── logs/ # Log files
├── notebooks/ # Jupyter notebooks for data exploration & model building
├── src/ # Source code modules
│ ├── pipeline.py # Pipeline logic (data prep, feature engineering, model training)
│ ├── api.py # API definitions (prediction endpoints)
│ ├── main.py # Entry point for launching the API
│ ├── mlflow_config.py # Configuration for MLflow (if used for tracking)
│ └── logging_config.py # Logging setup
├── tests/ # Unit tests
├── cbm_flight.pkl # Trained model serialized for inference
├── flight_price.ipynb # Notebook exploring data/training process
├── requirements.txt # Required packages
├── requirements-dev.txt # Dev dependencies (testing, linting, etc.)
└── pytest.ini # pytest configuration


---

## Getting Started

### Prerequisites

- Python 3.8+  
- `pip` for installing dependencies  
- (Optional) virtual environment tool like `venv` or `conda`  
- (Optional) MLflow if you're using tracking  

### Setup and Installation

```bash
# Clone the repo
git clone https://github.com/10mudassir007/Flight-Price-Prediction.git
cd Flight-Price-Prediction

# Checkout structured branch (if not already)
git checkout structured

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

```
### Running the API

```bash
uvicorn main:app
```