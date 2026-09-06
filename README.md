# 💳 Razorpay AI Fraud Detection & Risk Intelligence Dashboard

**AI-powered payment risk scoring, built for the Razorpay Buildathon 2026 — AI Risk Manager / Fraud Detection track.**

> Author: **Riddhi Pathak** — MA Economics (FinTech), Symbiosis College of Arts & Commerce (SPPU), Pune

![status](https://img.shields.io/badge/status-active-brightgreen) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![gradio](https://img.shields.io/badge/UI-Gradio-orange) ![model](https://img.shields.io/badge/model-Isolation%20Forest-072AC8)

--

---

## 🧭 Overview

Razorpay and other payment gateways process millions of transactions a day. This project is an end-to-end
**AI Risk Intelligence dashboard** that takes an anonymized, pre-scored transaction dataset and turns it into a
production-style risk operations tool:

- Detects anomalous/fraudulent transactions (Isolation Forest, pre-computed in the dataset).
- Classifies each transaction into **Approve / Manual Review / Block**.
- Surfaces risk trends, alerts, and model performance to a risk analyst in one screen.
- Includes a **live "what-if" risk checker** — enter an amount and hour, get a real-time risk score.

The dataset (`Razorpay_AI_Risk_Fraud_Model_Riddhi_Results.csv`, 19,355 rows × 37 columns) is a PCA-anonymized
transaction dataset (`V1`–`V28`) with model outputs already attached: `Fraud_Probability`, `Risk_Score`,
`Risk_Level`, `Action`, `Actual_Class`, `Predicted_Class`. **No merchant ID or payment-method column exists in
this dataset** — features referencing "merchant risk" in the Risk Operations panel are explicitly built as an
Hour-window proxy, not fabricated merchant data.

---

## ✨ Features

| Area | What it does |
|---|---|
| **KPI Header** | 8 premium KPI cards — totals, approval/fraud rate, avg risk score, blocked amount |
| **Analytics** | Hourly volume bar, decision donut, Hour×Risk heatmap, risk histogram, amount-vs-risk scatter, fraud trend line, top-15 risk bar |
| **AI Risk Operations** | Live fraud alerts, hour-cluster "high-risk segment" alert, velocity-fraud warning (time-gap based), off-hours monitoring, AI recommendation badges |
| **High-Risk Transactions** | Searchable, filterable (by Risk Level), sorted by Risk_Score, top 25 |
| **Executive Insights** | 10 business insights generated directly from the CSV at runtime (not hardcoded) |
| **Model Performance** | Confusion matrix, ROC curve + AUC, Precision/Recall/F1/Accuracy, Random Forest feature importance |
| **Live Risk Checker** | Isolation Forest fit on Hour + Scaled_Amount — score any hypothetical transaction live |
| **Raw Data** | Original results table (kept from the first version — nothing removed) |

---

## 🏗️ Architecture

```
CSV (pre-scored transactions)
        │
        ▼
Data cleaning (robust Action-label parsing: emoji/casing-safe)
        │
        ▼
┌───────────────┬─────────────────────┬───────────────────────────┐
│ KPI + Charts  │ Risk Ops / Alerts   │ Model Performance          │
│ (Plotly)      │ (rule-based, from   │ - Confusion Matrix /ROC    │
│               │  Actual_Class etc.) │   from Actual vs Predicted │
│               │                     │ - RandomForest fit live for│
│               │                     │   feature importance       │
└───────────────┴─────────────────────┴───────────────────────────┘
        │
        ▼
Gradio Blocks UI (tabs) ──▶ served via app.py
```

**Models used:**
1. **Isolation Forest** — the source of the dataset's `Risk_Score` / `Fraud_Probability` / `Action`, and re-used
   live in the "Live Risk Checker" tab (fit on `Hour` + `Scaled_Amount`).
2. **Random Forest Classifier** — trained at startup on `V1`–`V28` + `Scaled_Amount` + `Hour` against
   `Actual_Class`, used only to produce the **feature importance** chart in Model Performance.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Gradio** — dashboard UI
- **Plotly** — interactive charts
- **Pandas / NumPy** — data processing
- **scikit-learn** — Isolation Forest, Random Forest, metrics (confusion matrix, ROC/AUC, precision/recall/F1)

---

## 📂 Folder Structure

```
razorpay-ai-risk-dashboard/
├── app.py                     # Main Gradio dashboard (this project's entry point)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── Razorpay_AI_Risk_Fraud_Model_Riddhi_Results.csv   # Pre-scored transaction dataset
├── notebooks/
│   └── Razorpay_Risk_Dashboard.ipynb   # Original EDA / model notebook
├── screenshots/
│   ├── analytics_tab.png
│   ├── risk_ops_tab.png
│   ├── model_performance_tab.png
│   └── live_checker_tab.png
└── docs/
    ├── linkedin_post.md
    ├── resume_bullets.md
    └── buildathon_submission.md
```

---

## ⚙️ Installation (local)

```bash
git clone https://github.com/<your-username>/razorpay-ai-risk-dashboard.git
cd razorpay-ai-risk-dashboard

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

python app.py
```

The app will print a local URL (and a temporary public URL if `share=True` is set in `demo.launch()`).

---

## 🚀 Hugging Face Spaces Deployment

1. Create a new Space → **SDK: Gradio**.
2. Upload `app.py`, `requirements.txt`, and
   `Razorpay_AI_Risk_Fraud_Model_Riddhi_Results.csv` to the Space's root.
3. Hugging Face auto-installs `requirements.txt` and runs `app.py`.
4. No secrets or API keys are required — everything runs on the bundled CSV.
5. Once live, replace the screenshot placeholders above with real captures from the Space.

---

## 📈 Sample Executive Insights (generated live from the CSV)

The **Executive Insights** tab in the app computes these at runtime — a sample of what it surfaces:

1. Approval rate is dominant — the model is conservative, minimizing false declines.
2. The manual-review queue is very small relative to total volume.
3. Blocked transactions carry a meaningfully higher average scaled amount than approved ones.
4. Risk is concentrated in specific hour-windows rather than spread evenly.
5. A specific hour has both the highest block count and the highest average risk score.
6. Off-hours (00:00–05:00) traffic shows a disproportionately high block rate.
7. Model precision is strong — most blocked transactions are true fraud.
8. Recall has room to improve — some actual fraud is still scored as approved.
9. The risk-score distribution is heavily right-skewed — a small tail drives most exposure.
10. The dataset lacks merchant/payment-method columns, limiting segmentation depth.

---

## 🔭 Future Scope

- Add real **Merchant ID** and **Payment Method** fields to unlock true merchant-risk and channel-risk views.
- Replace the static Isolation Forest scores with a **live-retraining pipeline** on rolling transaction windows.
- Add **SHAP-based explainability** per transaction instead of only global feature importance.
- Wire the "Live Risk Checker" to a **real-time transaction stream** (Kafka/webhook) instead of manual sliders.
- Add **case management** (assign/resolve manual-review items) for a full risk-ops workflow.
- A/B test decision thresholds against business KPIs (approval rate vs fraud loss) before deployment.

---

## 🏁 Buildathon Submission Description

See [`docs/buildathon_submission.md`](docs/buildathon_submission.md).

## 💼 Resume Bullets

See [`docs/resume_bullets.md`](docs/resume_bullets.md).

## 📱 LinkedIn Post

See [`docs/linkedin_post.md`](docs/linkedin_post.md).

---

## 📄 License

Built as a personal portfolio / hackathon submission project. Not affiliated with or endorsed by Razorpay;
Razorpay-inspired styling is used for presentation purposes only.
