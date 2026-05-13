# PHISHGUARD AI – Explainable Real-Time Phishing Detection System

## Overview

PHISHGUARD AI is an AI-powered real-time phishing detection system designed to identify malicious and suspicious websites using machine learning, cybersecurity intelligence, and explainable risk analysis.

The system combines URL-based analysis, SSL verification, domain intelligence, redirect detection, homograph attack detection, and behavioral analysis to generate accurate phishing risk predictions with transparent explanations.

---

## Features

* Real-time phishing URL detection
* Machine Learning-based classification using Random Forest
* Explainable AI (XAI) risk interpretation
* URL feature extraction and heuristic analysis
* SSL certificate validation
* Domain intelligence and WHOIS analysis
* Redirect chain detection
* Homograph attack detection
* Interactive web interface using Flask
* Risk scoring and security insights dashboard

---

## Tech Stack

### Backend

* Python
* Flask
* Scikit-learn
* SQLite

### Frontend

* HTML
* CSS
* JavaScript
* Chart.js

### Security & Analysis

* WHOIS Intelligence
* SSL Verification
* Selenium Automation
* URL Heuristics
* Explainable AI

---

## System Architecture

1. User submits a URL through the web interface.
2. The system extracts URL and webpage features.
3. Domain intelligence and SSL validation are performed.
4. Redirect and homograph attack checks are executed.
5. The machine learning model evaluates phishing probability.
6. Explainable AI generates interpretable security insights.
7. Final phishing risk score and analysis are displayed.

---

## Core Detection Techniques

### URL Feature Analysis

* URL length analysis
* Suspicious keyword detection
* Special character identification
* IP-based URL detection
* Dot and subdomain analysis

### Domain Intelligence

* WHOIS lookup
* Domain age calculation
* Expiration analysis
* Reputation indicators

### SSL Verification

* HTTPS validation
* Certificate issuer verification
* SSL validity checks

### Advanced Security Analysis

* Redirect chain monitoring
* Homograph attack detection
* Login form analysis
* iFrame detection
* External link analysis

---

## Machine Learning Model

The phishing detection engine uses a Random Forest classifier trained on phishing and legitimate URL datasets.

### Model Capabilities

* Phishing classification
* Risk probability prediction
* Explainable output generation
* Feature importance analysis

---

## Project Structure

```bash
backend/
│── app.py
│── feature_extractor.py
│── ml_model.py
│── ssl_checker.py
│── domain_intelligence.py
│── redirect_detector.py
│── homograph_detector.py
│── explainable_ai.py

frontend/
│── templates/
│   ├── index.html
│   └── dashboard.html
│
│── static/
│   ├── style.css
│   └── script.js

database/
│── models.py

datasets/
│── phishing_dataset.csv

models/
│── phishing_model.pkl

requirements.txt
README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/PHISHGUARD-AI.git
cd PHISHGUARD-AI
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
python app.py
```

The application will run locally at:

```bash
http://127.0.0.1:5000
```

---

## Future Improvements

* Deep learning-based phishing detection
* Browser extension integration
* Real-time threat intelligence APIs
* Cloud deployment support
* Email phishing analysis
* API integration for SOC environments

---

## Use Cases

* Cybersecurity awareness platforms
* Security Operations Centers (SOC)
* Educational cybersecurity projects
* Phishing detection research
* Threat intelligence analysis

---

## Author

Sreelakshmi Reji



---

## License

This project is intended for educational and research purposes only.
