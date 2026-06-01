# Credit Risk Probability Model for Alternative Data

**Bati Bank × Xente eCommerce | KAIM Week 4 Challenge**

An end-to-end implementation for building, deploying, and automating a credit risk model using behavioral transaction data as a proxy for creditworthiness.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Credit Scoring Business Understanding](#credit-scoring-business-understanding)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Setup & Installation](#setup--installation)
6. [Usage](#usage)
7. [Results](#results)
8. [References](#references)

---

## Project Overview

Bati Bank is partnering with an eCommerce platform to enable a **buy-now-pay-later (BNPL)** service. Customers must be assessed for creditworthiness before being approved for credit purchases. The challenge is that the raw transaction data contains **no direct default label** — credit outcomes are not recorded in the eCommerce dataset. This project constructs a proxy credit risk target using RFM-based behavioral segmentation, then trains and deploys a scoring model that outputs a continuous risk probability for each applicant.

**Core deliverables:**
- A proxy target variable (`is_high_risk`) derived from RFM clustering
- A reproducible sklearn feature engineering pipeline
- Trained, tracked, and registered models via MLflow
- A containerized REST API (FastAPI + Docker) serving risk predictions
- A CI/CD pipeline (GitHub Actions) automating linting and testing

---

## Credit Scoring Business Understanding

### 1. How Does the Basel II Accord's Emphasis on Risk Measurement Influence the Need for Interpretable and Well-Documented Models?

The Basel II Capital Accord, formalized by the Bank for International Settlements, establishes a three-pillar framework for banking regulation: minimum capital requirements (Pillar 1), supervisory review (Pillar 2), and market discipline through public disclosure (Pillar 3). Within Pillar 1, banks that adopt the **Internal Ratings-Based (IRB) approach** are permitted to use their own models to estimate Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD) — the three components that determine regulatory capital requirements.

This framework has profound implications for model design:

**Interpretability as a Regulatory Requirement**
Basel II does not merely encourage interpretable models; it functionally requires them. Supervisory authorities must be able to validate that a bank's internal models are conceptually sound, empirically calibrated, and correctly implemented. A "black box" model that produces accurate aggregate predictions but cannot explain individual credit decisions fails this standard. Regulators need to audit the logic — not just the output — of risk models, which means that feature contributions, decision pathways, and score-to-probability mappings must be traceable and explainable.

**Documentation as a Compliance Artifact**
Under Pillar 2, supervisory review requires banks to demonstrate rigorous internal assessment processes. This translates into model documentation that covers: the statistical rationale for variable selection, evidence of discriminatory power (e.g., Gini coefficient, KS statistic), out-of-time and out-of-sample validation results, champion/challenger testing history, and a clear description of any proxy assumptions made in the absence of direct default labels. This documentation is reviewed by internal model validation teams and external regulators alike.

**Auditability and Model Monitoring**
Basel II also requires banks to monitor model performance over time and recalibrate when drift is detected. An interpretable model — particularly one built on Weight of Evidence (WoE) transformations — allows analysts to pinpoint exactly which input variable has drifted and update only that component, rather than retraining an opaque ensemble from scratch. This is operationally critical in production risk systems where stability, not just accuracy, is a business requirement.

In summary, the Basel II framework transforms model interpretability from a "nice-to-have" feature into a legal and operational necessity. Every modeling choice — from variable selection to score calibration — must be defensible to a regulator who may challenge it years after deployment.

---

### 2. Why Is a Proxy Variable Necessary When a Default Label Is Unavailable, and What Business Risks Does Proxy-Based Prediction Introduce?

**The Necessity of a Proxy Target**
Supervised machine learning for credit scoring requires a binary outcome variable: typically `1` (default) or `0` (no default). In traditional banking datasets, this label is derived from loan performance records — a borrower either missed contractual payments beyond a defined threshold (e.g., 90+ days past due, per Basel II's technical definition of default) or did not. The Xente eCommerce transaction dataset contains no such information: it records purchase behavior, not credit repayment history. There are no loan disbursements, no payment schedules, and no delinquency events in the data.

To apply supervised learning in this context, we must engineer a **proxy target variable** — a constructed binary label that approximates default risk using observable behavioral signals. The approach used here is RFM (Recency, Frequency, Monetary) segmentation: customers who are behaviorally disengaged (low transaction frequency, low total spend, long inactivity since last purchase) are hypothesized to represent a higher credit risk population. This hypothesis is grounded in behavioral economics research showing that financial disengagement and credit stress are correlated — customers who stop transacting may be experiencing financial difficulty.

The proxy label is created via K-Means clustering on normalized RFM scores, with the cluster exhibiting the worst RFM profile designated as `is_high_risk = 1`.

**Business Risks Introduced by Proxy-Based Prediction**

*Label validity risk:* The core assumption — that behavioral disengagement predicts loan default — may not hold in the Bati Bank context. A customer may be inactive on the eCommerce platform for entirely benign reasons (seasonal purchasing, switching platforms, travel) while being fully creditworthy. If the proxy systematically mislabels this group as high-risk, the model will produce false positives that deny credit to qualified applicants, resulting in lost revenue and potential claims of discriminatory lending.

*Circular reasoning and selection bias:* If the model trained on proxy labels is used to make actual lending decisions, and those decisions shape the population from which future training data is drawn, the model's assumptions become self-reinforcing. High-risk labels may reflect the bank's own past lending behavior rather than true credit risk, creating a feedback loop that is difficult to detect and correct.

*Regulatory and legal exposure:* In regulated markets, credit decisions must be explainable and fair. A proxy variable derived from behavioral clustering may inadvertently encode protected characteristics (e.g., geography, income level correlated with transaction frequency) that are illegal to use in credit decisions under anti-discrimination statutes. Without careful fairness auditing, the model could expose the bank to regulatory enforcement actions.

*Calibration mismatch:* Even if the model discriminates well between high- and low-risk proxy groups, the score it produces is not a true probability of default — it is a probability of belonging to a behaviorally disengaged cluster. Converting this to a credit decision requires an additional calibration step (e.g., Platt scaling, isotonic regression) and ongoing monitoring against realized default rates once actual loan outcomes become available.

*Model obsolescence:* Consumer behavior shifts over time, and the correlation between RFM patterns and credit risk is not static. A model trained on pre-lending behavioral data may degrade rapidly once BNPL is launched and customers begin optimizing their behavior specifically because they know they are being scored.

These risks do not make proxy-based modeling inadvisable — it is often the only viable path when direct labels are unavailable. But they require explicit disclosure in model documentation, conservative risk thresholds at deployment, and a plan to replace the proxy model with one trained on actual loan performance data as it accumulates.

---

### 3. What Are the Key Trade-offs Between Logistic Regression with WoE and Gradient Boosting in a Regulated Financial Environment?

The choice between a classical scorecard model and a modern ensemble method is one of the most consequential decisions in regulated credit modeling. The two approaches differ not just in predictive performance, but in interpretability, auditability, compliance posture, and operational maintainability.

**Logistic Regression with Weight of Evidence (WoE)**

Weight of Evidence transformation converts continuous and categorical predictors into a single numeric scale by computing `ln(Distribution of Events / Distribution of Non-Events)` for each bin of each variable. Information Value (IV) then quantifies each variable's predictive power, enabling principled variable selection. The resulting WoE-encoded features fed into Logistic Regression produce a model whose output is a linear combination of log-odds — directly interpretable as a scorecard.

*Advantages in a regulated context:*
- **Full transparency:** Each variable's contribution to the final score is a single coefficient multiplied by its WoE value. A credit officer can decompose any individual score and explain, in plain language, which factors drove a decision.
- **Basel II compliance by design:** The WoE-scorecard format maps directly onto the documentation requirements of Pillar 2. Regulators have decades of familiarity with this approach.
- **Stability:** Logistic Regression is less prone to overfitting on small datasets and tends to produce stable predictions when input distributions shift gradually.
- **Score interpretability:** The log-odds output can be scaled to a points-based scorecard (e.g., 300–850 range) that is intuitive to business users and compliant with adverse action notice requirements.

*Disadvantages:*
- **Assumes monotonic relationships:** WoE binning imposes an assumption that the risk relationship between a predictor and the target is monotonic within bins. Non-linear, interaction-dependent relationships are not captured.
- **Manual feature engineering overhead:** Optimal binning and WoE computation require domain expertise and iteration. Automation is possible but introduces its own risks.
- **Performance ceiling:** On complex, high-dimensional datasets with non-linear patterns, Logistic Regression typically underperforms ensemble methods in terms of AUC and Gini coefficient.

**Gradient Boosting (XGBoost / LightGBM)**

Gradient boosting builds an ensemble of decision trees sequentially, with each tree correcting the residual errors of its predecessors. The result is a highly flexible model capable of capturing complex non-linear relationships and feature interactions automatically.

*Advantages:*
- **Superior predictive performance:** Gradient boosting consistently achieves higher AUC scores on tabular credit data, particularly when the dataset is large and the feature set contains complex interactions.
- **Automatic feature interaction modeling:** No manual binning or WoE transformation is required; the algorithm discovers relevant interactions natively.
- **Handles missing values natively:** XGBoost and LightGBM have built-in missing value handling, reducing preprocessing requirements.

*Disadvantages in a regulated context:*
- **Interpretability deficit:** The ensemble of hundreds of trees has no natural analog to a scorecard. While post-hoc tools like SHAP (SHapley Additive exPlanations) provide feature-level attribution, these explanations are approximations, not the model itself. A regulator may question whether SHAP values constitute sufficient explainability for adverse action notices.
- **Overfitting risk:** Without careful regularization and cross-validation, gradient boosting can overfit to training data, particularly on datasets with a high feature-to-sample ratio.
- **Model governance complexity:** Versioning, monitoring, and recalibrating a gradient boosting model requires more sophisticated MLOps infrastructure than a scorecard. Changes to hyperparameters can have non-intuitive effects on output distributions.
- **Regulatory acceptance risk:** Some jurisdictions and regulators have not yet established clear guidance on the use of complex ML models for credit decisions. Deploying a gradient boosting model in a conservative regulatory environment may require additional validation burden.

**Practical Recommendation**
In a regulated financial environment, the optimal approach is typically a **two-model strategy**: use Logistic Regression with WoE as the production scoring model for compliance and explainability, and use Gradient Boosting as a benchmark to quantify the performance cost of interpretability. If the performance gap is small (e.g., AUC within 0.02–0.03), the interpretable model is clearly preferred. If the gap is large, it signals that important non-linear relationships exist in the data — which may justify investing in model explainability infrastructure (SHAP, LIME) to make the complex model regulatorily defensible, or engineering new features that allow the linear model to capture those relationships.

For this project, both models will be trained, tracked via MLflow, and evaluated on identical held-out test sets. The final deployment decision will be documented with reference to the Basel II considerations above.

---

## Dataset

The dataset is sourced from the [Xente Challenge on Kaggle](https://www.kaggle.com/datasets/atwine/xente-challenge). It contains transaction-level records from the Xente eCommerce platform.

| Field | Description |
|---|---|
| TransactionId | Unique transaction identifier |
| BatchId | Batch processing identifier |
| AccountId | Customer account identifier |
| SubscriptionId | Customer subscription identifier |
| CustomerId | Unique customer identifier |
| CurrencyCode | Transaction currency |
| CountryCode | Geographic country code |
| ProviderId | Source provider of purchased item |
| ProductId | Item purchased |
| ProductCategory | Broader product category |
| ChannelId | Transaction channel (web, Android, iOS, etc.) |
| Amount | Transaction value (positive = debit, negative = credit) |
| Value | Absolute transaction amount |
| TransactionStartTime | Transaction timestamp |
| PricingStrategy | Merchant pricing category |
| FraudResult | Fraud flag (1 = fraud, 0 = legitimate) |

Download data to `data/raw/` and add that directory to `.gitignore`.

---

## Project Structure

```
credit-risk-model/
├── .github/
│   └── workflows/
│       └── ci.yml                  # CI/CD pipeline (GitHub Actions)
├── data/                           # gitignored
│   ├── raw/                        # Raw Xente transaction data
│   └── processed/                  # Engineered features + is_high_risk label
├── notebooks/
│   └── eda.ipynb                   # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py          # Feature engineering pipeline
│   ├── train.py                    # Model training + MLflow tracking
│   ├── predict.py                  # Inference utilities
│   └── api/
│       ├── main.py                 # FastAPI application
│       └── pydantic_models.py      # Request/response schemas
├── tests/
│   └── test_data_processing.py     # Unit tests (pytest)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/credit-risk-model.git
cd credit-risk-model

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Place data/xente_data.csv in data/raw/
```

---

## Usage

```bash
# Run EDA notebook
jupyter notebook notebooks/eda.ipynb

# Run feature engineering pipeline
python src/data_processing.py

# Train models (with MLflow tracking)
python src/train.py

# Launch API locally
uvicorn src.api.main:app --reload

# Run with Docker
docker-compose up --build

# Run tests
pytest tests/
```

---

## Results

*(To be populated after model training — see Task 5 deliverables)*

---

## References

- [Basel II Capital Accord](https://fastercapital.com/content/Basel-Accords--What-They-Are-and-How-They-Affect-Credit-Risk-Management.html)
- [Alternative Credit Scoring — HKMA](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [Credit Scoring Approaches Guidelines — World Bank](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)
- [Weight of Evidence and Information Value](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)
- [Xente Challenge Dataset — Kaggle](https://www.kaggle.com/datasets/atwine/xente-challenge)