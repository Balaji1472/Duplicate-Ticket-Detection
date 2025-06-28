# 🌻 Duplicate Ticket Detection 

A powerful FastAPI-based application for detecting and managing duplicate support tickets using semantic similarity with transformer embeddings. Includes multilingual support, clustering, feedback collection, and a web dashboard.

---

## 🚀 Features

* 🔍 **Semantic Duplicate Detection** using Sentence Transformers
* 🌍 **Multilingual Support** with optional language detection
* 📦 **Batch Ticket Checking**
* 🧠 **Clustering** using KMeans (re-clusterable)
* 💬 **Feedback System** to collect active learning input
* 📊 **System Statistics API**
* 📺 **Streamlit Dashboard** for interactive UI
* 📜 Clean, testable, modular codebase

---

## 📂 Project Structure

```
duplicate_ticket_detector/
├── data/                 # Sample and feedback data
├── src/                  # Core modules (models, services, detectors)
├── tests/                # Pytest test cases
├── main.py               # FastAPI app entry point
├── dashboard.py          # Streamlit dashboard
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
https://github.com/Balaji1472/Duplicate-Ticket-Detection.git
cd Duplicate-Ticket-Detector
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# OR
source venv/bin/activate  # On Linux/macOS
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## ▶️ Running the API

```bash
python main.py
```

Access the docs:

* Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
* ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📊 Running the Dashboard

```bash
streamlit run dashboard.py
```

---

## 🧪 Running Tests

```bash
pytest
```

---

## 📝 Requirements (Unpinned)

```text
# Core API framework
fastapi
uvicorn
pydantic-settings

# Machine Learning and NLP
transformers
torch
scikit-learn
sentence-transformers

# Text processing
spacy
langdetect

# Data handling
python-multipart
numpy
pandas

# Development and testing
pytest
pytest-asyncio
httpx
```

> 💡 Also run: `python -m spacy download en_core_web_sm`

---

## 📌 Feedback & Contributions

Feel free to fork, improve, or suggest enhancements.
For issues, create a GitHub issue or contact: **[support@example.com](balajirama.2005@gmail.com)**

---

## 📄 License

MIT License – Use freely, contribute openly.
