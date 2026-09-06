# Razorpay Buildathon 2026 — Submission Description

**Track:** AI Risk Manager / Fraud Detection
**Project:** AI Fraud Detection & Risk Intelligence Dashboard
**Author:** Riddhi Pathak

Payment platforms process millions of transactions daily, and every one of them carries fraud risk that must
be assessed in milliseconds. This project builds a risk-operations dashboard around an Isolation Forest
anomaly-detection model, scoring 19,355 anonymized transactions and routing each into Approve, Manual
Review, or Block.

Beyond the base classification, the dashboard adds the layer a real risk team needs to act on it: eight live
KPI cards, seven interactive visualizations (hourly volume, decision distribution, an hour-by-risk heatmap,
amount-vs-risk scatter, and a fraud trend line), and a searchable, sortable high-risk transaction table. An AI
Risk Operations panel surfaces live fraud alerts, a velocity-fraud warning derived from transaction time-gaps,
and off-hours monitoring — all computed directly from the data rather than hardcoded.

Model performance is reported transparently: a confusion matrix and ROC curve benchmark predictions
against ground-truth labels, alongside precision, recall, F1, and accuracy. A Random Forest trained on the
transaction's anonymized features produces a feature-importance ranking, giving a business-readable view
into what drives the model's decisions. A live "what-if" checker lets a reviewer type in a transaction amount
and hour and get an instant risk score from the same model family.

The project is intentionally honest about its data's limits — it does not fabricate merchant IDs or payment
methods that aren't in the source CSV, instead using hour-of-day as an explicit proxy for segment-level risk,
with richer segmentation flagged as future work.
