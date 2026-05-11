# 🎓 Placement Probability Predictor

Predicts the probability of a college student getting placed based on academic 
and skill parameters using a custom-built Gaussian Naive Bayes classifier with 
PCA, served via FastAPI and Streamlit.

## 📌 Overview

This project takes 5 student input parameters and returns the probability of 
placement using a fully custom ML pipeline — no sklearn used. The model is 
built from scratch using NumPy, SciPy, and PyTorch-based Maximum Likelihood 
Estimation (MLE).

## 🧠 ML Pipeline
```
Raw Input → PCA (SVD) → PyTorch Gaussian MLE → Naive Bayes → Posterior Probability
```

1. **Data Preprocessing** — Categorical columns (Yes/No) mapped to binary (1/0). 
   Student ID column cleaned and converted to integer
2. **Feature Selection** — 5 numeric features selected based on correlation 
   analysis: IQ, Previous Semester Result, CGPA, Communication Skills, 
   Projects Completed
3. **PCA** — Manual implementation using Singular Value Decomposition (SVD) 
   to decorrelate features and reduce multicollinearity. Eigenvectors saved 
   as `eigen_vectors.npy`
4. **Gaussian MLE via PyTorch** — Custom neural network (`GaussianMLEstimatorNN`) 
   built in PyTorch that minimizes Negative Log-Likelihood (NLL) using SGD 
   optimizer to estimate μ (mean) and σ (std) per feature per class. This is 
   the gradient descent approach to MLE — mathematically equivalent to the 
   closed-form solution but demonstrates deep understanding of optimization
5. **Gaussian Naive Bayes** — Bayes' theorem used to compute the posterior 
   probability of placement with a normalizing constant for calibrated 
   true probabilities

## 🔬 How the Math Works

**PCA:** Covariance matrix → SVD → Eigenvectors project input into 
decorrelated space

**Gaussian NLL Loss (PyTorch MLE):**
```
L(μ, σ) = -mean( log( (1/√(2πσ²)) × exp(-(x-μ)²/2σ²) ) )
```

**Bayes' Theorem:**
```
P(placed | features) = P(features | placed) × P(placed) / P(features)
```

**Normalizing Constant:**
```
P(features) = P(features|placed=1) × P(placed=1) 
            + P(features|placed=0) × P(placed=0)
```

**Gaussian Likelihood per feature:**
```
P(xᵢ | class) = (1 / √(2πσ²)) × exp(-(xᵢ - μ)² / 2σ²)
```

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | NumPy, SciPy, Pandas, PyTorch |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit |
| Notebook | Jupyter |
| Environment | Python 3.12, Ubuntu 24.04 (WSL) |

## 📂 Project Structure
```
placement-probability-predictor/
├── ds-project-1.ipynb                  # ML pipeline notebook
├── backend.py                          # FastAPI REST API
├── streamlit_app.py                    # Streamlit frontend
├── config.py                           # Prior probability and feature constants
├── eigen_vectors.npy                   # Saved PCA eigenvectors
├── likelihood_distribution_params.pkl  # Saved Gaussian MLE parameters
├── college_student_placement_dataset.csv  # Dataset
├── requirements.txt                    # Python dependencies
└── README.md
```

## ⚙️ How to Run

**1. Clone the repo**
```bash
git clone https://github.com/Divyanshusinghrajawat/placement-probability-predictor.git
cd placement-probability-predictor
```

**2. Create virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**4. Run the backend (Terminal 1)**
```bash
uvicorn backend:app --reload
```

**5. Run the frontend (Terminal 2)**
```bash
streamlit run streamlit_app.py
```

**6. Open in browser**
```
http://localhost:8501
```

## 📊 Dataset

- **Source:** College Student Placement Dataset
- **Rows:** Custom dataset
- **Features:** IQ, Previous Semester Result, CGPA, Communication Skills, 
  Projects Completed, Placement Status
- **Target:** Placed (0 or 1)
- **Prior P(placed=1):** 0.1659 (~16.6% placement rate)

## 🔍 Input Features

| Feature | Description | Range |
|---|---|---|
| IQ | Intelligence Quotient score | 40 – 160 |
| Previous Semester Result | Last semester score out of 10 | 0.0 – 10.0 |
| CGPA | Cumulative Grade Point Average | 0.0 – 10.0 |
| Communication Skills | Self-rated communication score | 0 – 10 |
| Projects Completed | Number of projects done | 0 – 5 |

## 🏗️ Architecture
```
┌─────────────────┐         HTTP POST          ┌──────────────────────┐
│                 │   /compute_probability      │                      │
│  Streamlit UI   │ ─────────────────────────► │   FastAPI Backend    │
│  (port 8501)    │ ◄───────────────────────── │   (port 8000)        │
│                 │      JSON response          │                      │
└─────────────────┘                            └──────────────────────┘
                                                          │
                                               ┌──────────▼───────────┐
                                               │   ML Inference       │
                                               │  eigen_vectors.npy   │
                                               │  params.pkl          │
                                               └──────────────────────┘
```

## 💡 Key Design Decisions

- **PyTorch for MLE** — Used gradient descent (SGD + NLL loss) to estimate 
  Gaussian parameters instead of the closed-form solution, to demonstrate 
  understanding of optimization and loss minimization
- **PCA from scratch** — Used NumPy SVD directly instead of sklearn's PCA 
  to show understanding of the underlying linear algebra
- **Normalized posterior** — Backend computes true posterior probabilities 
  using the normalizing constant, not just unnormalized likelihoods
- **Client-server architecture** — Streamlit frontend and FastAPI backend 
  run as separate services communicating via REST API, mirroring 
  production ML deployment patterns

## 👨‍💻 Author

**Divyanshu Singh Rajawat**
B.Tech Computer Science | Poornima University
[GitHub](https://github.com/Divyanshusinghrajawat)# placement-probability-predictor-main
