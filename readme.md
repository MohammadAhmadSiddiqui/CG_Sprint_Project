#  DiabetesAI — Healthcare Multi-Agent AI Platform

**Left Shift Program 2026 – Data & AI (T5) | Capstone Project**

A production-grade, end-to-end Multi-Agent AI Platform built for the **Intelligent Healthcare Support System** domain. The platform integrates Machine Learning, Generative AI, RAG, multi-agent orchestration, Azure cloud services, and a deployed web application.


---

##  Project Overview

This platform assists healthcare providers with:
- **Diabetes risk prediction** using a trained Random Forest model
- **Medical knowledge Q&A** using RAG over a curated knowledge base
- **Dataset analytics** using an AI-powered Data Analyst Agent
- **Patient data ingestion** into Azure SQL database
- **Multi-agent orchestration** that automatically routes queries to the right agent

---

##  Architecture

```
User (Browser)
      ↓
Frontend (index.html — DiabetesAI Dashboard)
      ↓
FastAPI Backend (app.py)
      ↓
┌─────────────────────────────────────────────┐
│           Multi-Agent Orchestrator           │
│  Routes queries using Azure OpenAI (GPT)    │
└──────────┬──────────────┬───────────────────┘
           ↓              ↓              ↓
  Document Agent    Data Analyst    ML Expert Agent
  (RAG + FAISS)     Agent (CSV)    (Random Forest)
           ↓              ↓              ↓
  Azure OpenAI     Azure OpenAI    Azure OpenAI
  Embeddings       GPT-4.1-mini    GPT-4.1-mini
           ↓
      FAISS Vector Store
      (knowledge_base/)
                              ↓
                        Azure SQL
                    (patient_records)
```

---

##  Project Structure

```
New Sprint/
├── app.py                          # FastAPI backend — 4 REST APIs
├── index.html                      # Frontend dashboard
├── requirements.txt                # Python dependencies
├── stratup.sh                      # Azure startup command
├── .env                            # Local environment variables (not committed)
│
├── agents/
│   ├── orchestrator.py             # Multi-agent router (LLM-based)
│   ├── document_assistant_agent.py # RAG agent — FAISS + embeddings
│   ├── data_analyst_agent.py       # Dataset analytics agent
│   └── ml_expert_agent.py          # ML prediction + LLM explanation
│
├── knowledge_base/
│   └── diabetes_medical_info.txt   # Medical facts for RAG
│
├── vector_store/
│   ├── index.faiss                 # FAISS vector index
│   └── index.pkl                   # FAISS metadata
│
├── models/
│   ├── diabetes_model.pkl          # Trained Random Forest model
│   ├── gender_encoder.pkl          # Label encoder — gender
│   └── smoking_encoder.pkl         # Label encoder — smoking history
│
├── dataset/
│   ├── diabetes_prediction_dataset.csv   # Raw dataset (100,000 records)
│   └── processed/
│       ├── cleaned_data.csv              # After data cleaning
│       └── engineered_data.csv           # After feature engineering
│
├── nootbook/
│   ├── data_cleaning.ipynb         # Step 1 — Data cleaning pipeline
│   ├── feature_engineering.ipynb   # Step 2 — Feature engineering
│   └── model_training_evaluation.ipynb  # Step 3 — Model training & evaluation
│
├── tests/
│   └── test_app.py                 # 14 pytest unit tests
│
└── .github/
    └── workflows/
        └── deploy.yml              # CI/CD — GitHub Actions pipeline
```

---

##  Setup & Installation

### Prerequisites
- Python 3.11+
- ODBC Driver 18 for SQL Server → https://aka.ms/odbc18
- Azure CLI → https://aka.ms/installazurecli
- Azure account with Azure AI Foundry access

### 1. Clone the repository
```bash
git clone https://github.com/MohammadAhmadSiddiqui/CG_Sprint_Project.git
cd "CG_Sprint_Project/New Sprint"
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root:
```dotenv
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.services.ai.azure.com/openai/v1/
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_API_VERSION=2025-04-14
AZURE_SQL_SERVER=your-server.database.windows.net
AZURE_SQL_DATABASE=DataIngestion
```

### 5. Login to Azure (for SQL authentication)
```bash
az login
```

### 6. Run the application
```bash
uvicorn app:app --reload
```

Open: http://127.0.0.1:8000

---

##  Inspecting the Vector Store

To view the contents of the FAISS vector store (document chunks + embeddings):

**Basic view** — shows all document chunks + vector info:
```bash
cd vector_store
python view_vector_store.py --store-dir "."
```

**Detailed view** — also shows first 10 dimensions of each embedding vector:
```bash
cd vector_store
python view_vector_store.py --store-dir "." --show-vectors
```

This shows:
- All document chunks stored in the knowledge base
- Number of vectors, embedding dimensions (1536-dim)
- L2 magnitude of each embedding vector
- (with `--show-vectors`) First 10 float values of each embedding

##  API Endpoints

| # | Method | Endpoint | Description |
|---|--------|----------|-------------|
| 1 | POST | `/api/ingest` | Store patient record in Azure SQL |
| 2 | POST | `/api/predict` | Diabetes risk prediction (ML model) |
| 3 | POST | `/api/search` | RAG-based medical knowledge Q&A |
| 4 | POST | `/api/agent` | Multi-agent orchestrator |
| - | GET | `/health` | Health check |
| - | GET | `/` | Frontend dashboard |
| - | GET | `/docs` | Swagger UI |

### Example — Predict Diabetes Risk
```bash
curl -X POST https://healthcare-ai-platform.azurewebsites.net/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "age": 62,
    "hypertension": 1,
    "heart_disease": 1,
    "smoking_history": "current",
    "bmi": 35.0,
    "HbA1c_level": 7.8,
    "blood_glucose_level": 200
  }'
```

### Example — Medical Knowledge Search (RAG)
```bash
curl -X POST https://healthcare-ai-platform.azurewebsites.net/api/search \
  -H "Content-Type: application/json" \
  -d '{"question": "What HbA1c level indicates diabetes?"}'
```

### Example — Multi-Agent Orchestrator
```bash
curl -X POST https://healthcare-ai-platform.azurewebsites.net/api/agent \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What percentage of patients in the dataset have diabetes?",
    "patient_data": null
  }'
```

---

##  Multi-Agent System

| Agent | File | Responsibility |
|-------|------|----------------|
|  **Orchestrator** | `orchestrator.py` | Routes queries to the right agent using LLM |
|  **Document Assistant** | `document_assistant_agent.py` | RAG over medical knowledge base |
|  **Data Analyst** | `data_analyst_agent.py` | Answers dataset statistics questions |
|  **ML Expert** | `ml_expert_agent.py` | Runs prediction + generates LLM explanation |

### How Routing Works
```
User Query → Orchestrator (GPT-4.1-mini classifies intent)
                ↓
    "What is HbA1c?"     → Document Agent (RAG)
    "Average BMI?"        → Data Analyst Agent
    patient_data provided → ML Expert Agent
```

---

##  Machine Learning Model

| Item | Detail |
|------|--------|
| Algorithm | Random Forest Classifier |
| Dataset | Diabetes Prediction Dataset (100,000 records) |
| Features | age, gender, BMI, HbA1c, blood glucose, hypertension, heart disease, smoking history, bmi_category, glucose_risk |
| Engineered Features | `bmi_category` (0–3), `glucose_risk` (0/1) |
| Persistence | `models/diabetes_model.pkl` (pickle) |

### Data Pipeline
```
Raw CSV (100K records)
    ↓ data_cleaning.ipynb
Cleaned Data (null handling, encoding)
    ↓ feature_engineering.ipynb
Engineered Data (bmi_category, glucose_risk added)
    ↓ model_training_evaluation.ipynb
Trained Random Forest Model → diabetes_model.pkl
```

---

##  Azure Services Used

| Service | Purpose |
|---------|---------|
| **Azure AI Foundry** | GPT-4.1-mini (chat) + text-embedding-3-small (embeddings) |
| **Azure Web App (B1)** | Hosts FastAPI backend + frontend |
| **Azure SQL Database** | Stores patient records (DataIngestion database) |
| **GitHub Actions** | CI/CD pipeline — test + deploy |
| **Azure Managed Identity** | Passwordless SQL authentication |

---

##  Running Tests

```bash
pytest tests/test_app.py -v
```

**14 tests covering:**
- Health check
- Data ingestion (valid + missing fields)
- ML prediction (diabetic + non-diabetic + missing fields)
- Document search / RAG (2 questions + edge case)
- Agent routing (document, data, ML agents)
- ML model unit tests (high risk + low risk)

---

##  CI/CD Pipeline

Located at `.github/workflows/deploy.yml`

```
Push to main branch
        ↓
Job 1: Run 14 pytest tests
        ↓ (only if all pass)
Job 2: Deploy to Azure Web App
        ↓
https://healthcare-ai-platform.azurewebsites.net 
```

**GitHub Secrets required:**
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_SQL_SERVER`
- `AZURE_SQL_DATABASE`
- `AZURE_WEBAPP_PUBLISH_PROFILE`

---

##  Security

- API keys stored as **environment variables** (never hardcoded)
- Azure SQL uses **Managed Identity** authentication (no username/password)
- GitHub Secrets used for CI/CD credentials
- CORS configured for API access control

---


##  Author

**Mohammad Ahmad Siddiqui**  