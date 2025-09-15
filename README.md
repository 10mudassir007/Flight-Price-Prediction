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
```
Flight-Price-Prediction/
├── README.md
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── main.py
├── main-base.py
├── api.py
├── pipeline.py
├── mlflow_config.py
├── logging_config.py
├── data/
│   ├── raw_data.csv
│   └── processed_data.csv
├── dist/
│   ├── favicon.ico
│   ├── index.html
│   ├── placeholder.svg
│   ├── robots.txt
│   └── assets/
├── logs/
│   └── project.log
├── notebooks/
│   ├── flight_price.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── predict.py
│   └── utils.py
└── tests/
    ├── test_data.py
    ├── test_geoapi.py
    └── test_models.py

```
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

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
pip install -r requirements-dev.txt  # optional, for testing and dev tools
```

### Run the tests
```bash
pytest tests/
```


### Run the pipeline

```bash
python pipeline.py

```
### Running the API

```bash
uvicorn main:app
```
