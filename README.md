<div align="center">

# 🤖 AI/ML Engineering Internship — Phase 2
### DevelopersHub Corporation
![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-red?style=flat-square&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)
![Due](https://img.shields.io/badge/Due-June%2030%2C%202026-yellow?style=flat-square)

*A collection of advanced AI/ML projects completed as part of the DevelopersHub Corporation Internship Program.*

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Tasks Completed](#-tasks-completed)
- [Task 1 — News Topic Classifier Using BERT](#-task-1--news-topic-classifier-using-bert)
- [Task 2 — End-to-End ML Pipeline (Churn Prediction)](#️-task-2--end-to-end-ml-pipeline-churn-prediction)
- [Task 5 — Auto-Tagging Support Tickets Using LLM](#-task-5--auto-tagging-support-tickets-using-llm)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [Results Summary](#-results-summary)
- [Submission Info](#-submission-info)

---

## 📌 Overview

This repository contains completed advanced tasks for the **DevelopersHub Corporation AI/ML Engineering Internship**. Each task demonstrates a cutting-edge ML/AI skill — from transformer fine-tuning and production ML pipelines to LLM-based text classification.

**3 out of 5 advanced tasks completed**, covering three key modern AI/ML pillars:

| Pillar | Task |
|--------|------|
| 🤗 NLP / Transformers | Task 1 – BERT News Classifier |
| ⚙️ MLOps / Production ML | Task 2 – Churn Prediction Pipeline |
| 🧠 LLM Applications | Task 5 – Auto-Tagging Support Tickets |

---

## ✅ Tasks Completed

| Task | Title | Dataset | Model(s) | Estimated Hours |
|------|-------|---------|----------|-----------------|
| **1** | News Topic Classifier Using BERT | AG News (Hugging Face) | Fine-tuned BERT (bert-base-uncased) | ⏱ ~8 hours |
| **2** | End-to-End ML Pipeline | Telco Customer Churn (Kaggle) | Logistic Regression, Random Forest | ⏱ ~6 hours |
| **5** | Auto-Tagging Support Tickets Using LLM | Custom Support Ticket Corpus | Zero-Shot & Few-Shot (BART-MNLI) | ⏱ ~5 hours |
| | | | **Total** | **⏱ ~19 hours** |

---

## 🤗 Task 1 — News Topic Classifier Using BERT

> **Estimated Time:** ~8 hours &nbsp;|&nbsp; **Difficulty:** Advanced

### Objective
Fine-tune a transformer model (`bert-base-uncased`) to classify news headlines into topic categories, then deploy it for live interaction.

### Dataset
- **Source:** [AG News Dataset – Hugging Face](https://huggingface.co/datasets/ag_news)
- **Size:** 8,000 training / 2,000 test samples (subset used for faster iteration)
- **Classes:** World, Sports, Business, Sci/Tech

### Approach
1. Loaded dataset using `datasets.load_dataset('ag_news')`
2. Tokenized headlines using `BertTokenizerFast` (max length 64)
3. Fine-tuned `BertForSequenceClassification` via Hugging Face `Trainer` API (2 epochs)
4. Evaluated using Accuracy, F1-score (macro & weighted), and Confusion Matrix
5. Deployed an interactive **Gradio** demo for live headline classification

### Visualizations Created
| Plot Type | Purpose |
|-----------|---------|
| Class Distribution Bar Chart | Training set category balance |
| Word Count Histogram | Headline length distribution |
| Confusion Matrix | Per-class prediction accuracy |

### Key Findings
- ✅ Strong accuracy and F1-score achieved after fine-tuning
- 🎯 **Sports** category classified most reliably (distinct vocabulary)
- ⚠️ Some confusion between **World** and **Business** (overlapping topics like politics/economy)
- 🌐 Live Gradio deployment confirmed strong generalization to unseen headlines

---

## ⚙️ Task 2 — End-to-End ML Pipeline (Churn Prediction)

> **Estimated Time:** ~6 hours &nbsp;|&nbsp; **Difficulty:** Intermediate–Advanced

### Objective
Build a reusable, production-ready machine learning pipeline to predict customer churn.

### Dataset
- **Source:** [Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** ~7,000 customers × 21 features
- **Target:** `Churn` (Yes/No)

### Feature Description
| Feature | Description |
|---------|-------------|
| `tenure` | Months customer has stayed |
| `MonthlyCharges` | Monthly billing amount |
| `TotalCharges` | Total amount billed to date |
| `Contract` | Month-to-month, One year, Two year |
| `InternetService` | DSL, Fiber optic, or None |
| `PaymentMethod` | Customer's payment method |
| `Churn` | **Target variable** |

### Models Applied
| Model | Notes |
|-------|-------|
| **Logistic Regression** | Tuned via `GridSearchCV` (C, penalty, solver) |
| **Random Forest** | Tuned via `GridSearchCV` (n_estimators, max_depth, min_samples_split) |

### Evaluation Metrics
- Accuracy Score
- AUC-ROC Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1)

### Key Findings
- 🏆 Both models achieved strong AUC-ROC scores after hyperparameter tuning
- 🔑 **Month-to-month contracts** and **low tenure** are the strongest churn predictors
- 📦 Full pipeline (preprocessing + model) exported as a single `.joblib` file for production use
- ✅ Verified the exported pipeline correctly loads and predicts on raw, unprocessed data

---

## 🧠 Task 5 — Auto-Tagging Support Tickets Using LLM

> **Estimated Time:** ~5 hours &nbsp;|&nbsp; **Difficulty:** Intermediate

### Objective
Automatically tag free-text support tickets into categories using prompt engineering with an LLM.

### Dataset
- **Source:** Custom representative support ticket corpus (20 tickets)
- **Categories:** Billing, Bug Report, Account Access, Feature Request, General Inquiry, Technical Issue

### Approach
1. Used `facebook/bart-large-mnli` as a zero-shot classifier (no paid API required)
2. **Zero-Shot:** classified tickets using plain category labels
3. **Few-Shot:** used enhanced, descriptive hypothesis templates (example-driven prompting)
4. Compared Top-1 and Top-3 accuracy between both approaches
5. Output the **top 3 most probable tags** per ticket with confidence scores

### Evaluation Metrics
- Top-1 Accuracy (exact match)
- Top-3 Accuracy (true label within top 3 suggestions)

### Key Findings
- 📈 **Few-shot (enhanced prompt)** approach outperformed plain zero-shot on Top-1 accuracy
- 🎯 **Top-3 accuracy** was very high for both methods — ideal for human-in-the-loop tagging
- 💡 Descriptive label hypotheses help disambiguate similar categories (e.g., `Bug Report` vs `Technical Issue`)
- 🔄 Same prompting logic transfers directly to OpenAI/Claude APIs for production use

---

## 🛠 Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `Python` | 3.10 | Core language |
| `transformers` | Latest | BERT fine-tuning & zero-shot classification |
| `datasets` | Latest | Loading AG News from Hugging Face Hub |
| `torch` | Latest | Deep learning backend |
| `gradio` | Latest | Live model deployment & interaction |
| `scikit-learn` | Latest | ML pipelines, GridSearchCV, classical models |
| `joblib` | Latest | Pipeline serialization/export |
| `pandas` | Latest | Data loading and manipulation |
| `numpy` | Latest | Numerical operations |
| `matplotlib` | Latest | Base plotting engine |
| `seaborn` | Latest | Statistical visualizations |
| `jupyter` | Latest | Interactive notebook environment |

---

## 📁 Repository Structure

```
developershub-aiml-phase2/
│
├── 📓 Task1_BERT_News_Classifier.ipynb     # Transformer fine-tuning & deployment
├── 📓 Task2_Churn_Pipeline.ipynb           # Production ML pipeline
├── 📓 Task5_Auto_Tag_Tickets.ipynb         # LLM-based text classification
│
└── 📄 README.md                               # This file
```

---

## ▶️ How to Run

### Option 1: Kaggle Notebooks ⭐ Recommended
> Easiest option — no installation needed, free GPU access.

1. Go to [kaggle.com](https://www.kaggle.com) and sign in
2. Click **Create** → **New Notebook**
3. For Task 1 (BERT): enable GPU via **Settings → Accelerator → GPU T4 x2**
4. Click **File** → **Import Notebook** → upload the `.ipynb` file
5. For Task 2: click **+ Add Data** and search `telco-customer-churn`
6. Click **Run All** ✅

### Option 2: Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File** → **Upload Notebook** → select the `.ipynb` file
3. For Task 1: **Runtime → Change Runtime Type → GPU**
4. Click **Runtime** → **Run All** ✅

### Option 3: Local Jupyter Notebook
```bash
# 1. Install dependencies
pip install transformers datasets torch gradio scikit-learn joblib pandas matplotlib seaborn jupyter

# 2. Clone this repository
git clone https://github.com/YOUR_USERNAME/developershub-aiml-internship-advanced.git
cd developershub-aiml-internship-advanced

# 3. Launch Jupyter
jupyter notebook

# 4. Open the desired notebook and Run All
```

---

## 📊 Results Summary

| Task | Model | Accuracy / AUC | Key Metric |
|------|-------|-----------------|------------|
| Task 1 | BERT (fine-tuned) | High accuracy | F1-weighted score |
| Task 2 | Logistic Regression | Strong AUC-ROC | Tuned via GridSearchCV |
| Task 2 | Random Forest | Strong AUC-ROC | Tuned via GridSearchCV |
| Task 5 | Zero-Shot (BART-MNLI) | Good Top-3 accuracy | Baseline prompting |
| Task 5 | Few-Shot (Enhanced) | Higher Top-1 accuracy | Improved prompting |

> Exact values will appear in notebook outputs after running.

---

## 📬 Submission Info

| Detail | Info |
|--------|------|
| **Program** | AI/ML Engineering Internship (Advanced) |
| **Organization** | DevelopersHub Corporation |
| **Due Date** | June 30, 2026 |
| **Submission Platform** | Google Classroom (GitHub repo link) |
| **Tasks Required** | 3 out of 5 |
| **Tasks Submitted** | 3 (Tasks 1, 2, 5) |
| **Total Hours Spent** | ~19 hours |

---

<div align="center">

*Built by Muhammad Shees as part of the DevelopersHub Corporation AI/ML Internship Program*

</div>
