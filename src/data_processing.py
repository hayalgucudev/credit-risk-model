"""
data_processing.py
==================
Full feature-engineering + proxy-target pipeline for the Bati Bank
credit-risk model.  Produces a model-ready DataFrame from raw Xente
transaction data.

Pipeline steps
--------------
1. Aggregate per-customer transaction features
2. Extract temporal features
3. Compute RFM metrics
4. K-Means cluster customers → is_high_risk proxy label
5. Encode categoricals (one-hot)
6. Impute any residual missing values (median strategy)
7. Scale numerical features (StandardScaler)

Usage
-----
    python src/data_processing.py \
        --input  data/raw/data.csv \
        --output data/processed/processed.csv
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
SNAPSHOT_DATE = pd.Timestamp("2019-02-14", tz="UTC")
N_CLUSTERS = 3


# ── Custom transformers ────────────────────────────────────────────────────

class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Aggregates transaction-level rows into one row per customer.

    Produces:
        total_amount, avg_amount, txn_count, std_amount,
        max_amount, min_amount, debit_ratio, fraud_txn_count,
        unique_products, unique_categories, unique_channels
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["is_debit"] = (df["Amount"] > 0).astype(int)

        agg = df.groupby("CustomerId").agg(
            total_amount=("Amount", "sum"),
            avg_amount=("Amount", "mean"),
            txn_count=("TransactionId", "count"),
            std_amount=("Amount", "std"),
            max_amount=("Amount", "max"),
            min_amount=("Amount", "min"),
            debit_ratio=("is_debit", "mean"),
            fraud_txn_count=("FraudResult", "sum"),
            unique_products=("ProductId", "nunique"),
            unique_categories=("ProductCategory", "nunique"),
            unique_channels=("ChannelId", "nunique"),
        ).reset_index()

        agg["std_amount"] = agg["std_amount"].fillna(0)
        logger.info("AggregateFeatures: %d customers", len(agg))
        return agg


class TemporalFeatures(BaseEstimator, TransformerMixin):
    """
    Extracts hour, day, month, year from TransactionStartTime and
    re-merges the first-seen values back to the customer-level frame
    produced by AggregateFeatures.

    Expects the *original* transaction DataFrame to be passed in;
    the aggregated customer frame is returned with temporal columns.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["TransactionStartTime"] = pd.to_datetime(
            df["TransactionStartTime"], utc=True
        )
        df["txn_hour"] = df["TransactionStartTime"].dt.hour
        df["txn_day"] = df["TransactionStartTime"].dt.day
        df["txn_month"] = df["TransactionStartTime"].dt.month
        df["txn_year"] = df["TransactionStartTime"].dt.year

        temporal = (
            df.groupby("CustomerId")[["txn_hour", "txn_day", "txn_month", "txn_year"]]
            .agg(
                avg_txn_hour=("txn_hour", "mean"),
                avg_txn_day=("txn_day", "mean"),
                modal_month=("txn_month", lambda x: x.mode().iloc[0]),
                modal_year=("txn_year", lambda x: x.mode().iloc[0]),
            )
            .reset_index()
        )

        logger.info("TemporalFeatures: extracted for %d customers", len(temporal))
        return temporal


class RFMFeatures(BaseEstimator, TransformerMixin):
    """
    Computes Recency, Frequency, Monetary per customer from raw transactions.
    Returns a customer-level DataFrame with RFM columns.
    """

    def __init__(self, snapshot_date=SNAPSHOT_DATE):
        self.snapshot_date = snapshot_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["TransactionStartTime"] = pd.to_datetime(
            df["TransactionStartTime"], utc=True
        )

        rfm = df.groupby("CustomerId").agg(
            recency=("TransactionStartTime",
                      lambda x: (self.snapshot_date - x.max()).days),
            frequency=("TransactionId", "count"),
            monetary=("Amount", lambda x: x[x > 0].sum()),
        ).reset_index()

        logger.info(
            "RFMFeatures: recency mean=%.1f, freq median=%.0f, monetary median=%.0f",
            rfm["recency"].mean(),
            rfm["frequency"].median(),
            rfm["monetary"].median(),
        )
        return rfm


class ProxyTargetEngineer(BaseEstimator, TransformerMixin):
    """
    Clusters customers on their RFM profile using K-Means (k=3)
    and assigns is_high_risk = 1 to the cluster with the lowest
    engagement (lowest frequency + lowest monetary value).

    Adds column  is_high_risk  to the DataFrame passed in.
    The DataFrame must already contain  recency, frequency, monetary.
    """

    def __init__(self, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_ = None
        self.scaler_ = None
        self.high_risk_cluster_ = None

    def fit(self, X, y=None):
        rfm_cols = ["recency", "frequency", "monetary"]
        rfm = X[rfm_cols].copy()

        self.scaler_ = StandardScaler()
        rfm_scaled = self.scaler_.fit_transform(rfm)

        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = self.kmeans_.fit_predict(rfm_scaled)

        # Identify the high-risk cluster: lowest frequency + lowest monetary
        cluster_profiles = X.copy()
        cluster_profiles["cluster"] = labels
        cluster_summary = cluster_profiles.groupby("cluster")[
            ["frequency", "monetary"]
        ].mean()
        # Score each cluster: lower is worse (higher risk)
        cluster_summary["risk_score"] = (
            cluster_summary["frequency"].rank() + cluster_summary["monetary"].rank()
        )
        self.high_risk_cluster_ = int(cluster_summary["risk_score"].idxmin())
        logger.info(
            "ProxyTarget: high-risk cluster = %d | centroids:\n%s",
            self.high_risk_cluster_,
            cluster_summary.round(1).to_string(),
        )
        return self

    def transform(self, X):
        rfm_cols = ["recency", "frequency", "monetary"]
        rfm_scaled = self.scaler_.transform(X[rfm_cols])
        labels = self.kmeans_.predict(rfm_scaled)
        out = X.copy()
        out["cluster"] = labels
        out["is_high_risk"] = (out["cluster"] == self.high_risk_cluster_).astype(int)
        out = out.drop(columns=["cluster"])
        high_risk_rate = out["is_high_risk"].mean() * 100
        logger.info(
            "ProxyTarget: %.1f%% of customers labelled high-risk", high_risk_rate
        )
        return out


class CategoricalModeAggregator(BaseEstimator, TransformerMixin):
    """
    Picks the modal category per customer for ProductCategory,
    ChannelId, ProviderId, PricingStrategy.
    Returns a customer-level DataFrame.
    """

    CAT_COLS = ["ProductCategory", "ChannelId", "ProviderId", "PricingStrategy"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg = (
            X.groupby("CustomerId")[self.CAT_COLS]
            .agg(lambda x: x.mode().iloc[0])
            .reset_index()
        )
        logger.info("CategoricalModeAggregator: done for %d customers", len(agg))
        return agg


# ── Public pipeline builder ────────────────────────────────────────────────

def build_processing_pipeline():
    """
    Returns an unfitted ProxyTargetEngineer (the last step that
    produces is_high_risk).  The full processing is orchestrated
    by  run_pipeline()  below because each transformer operates
    on the raw transaction frame and we need to join results.
    """
    return ProxyTargetEngineer(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)


def run_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Full end-to-end processing: raw CSV → model-ready CSV.

    Returns the processed DataFrame.
    """
    logger.info("Loading raw data from %s", input_path)
    raw = pd.read_csv(input_path)
    raw["TransactionStartTime"] = pd.to_datetime(raw["TransactionStartTime"], utc=True)
    logger.info("Raw shape: %s", raw.shape)

    # 1. Aggregate features
    agg_transformer = AggregateFeatures()
    customer_agg = agg_transformer.transform(raw)

    # 2. Temporal features
    temporal_transformer = TemporalFeatures()
    customer_temporal = temporal_transformer.transform(raw)

    # 3. RFM features
    rfm_transformer = RFMFeatures(snapshot_date=SNAPSHOT_DATE)
    customer_rfm = rfm_transformer.transform(raw)

    # 4. Categorical modal values
    cat_transformer = CategoricalModeAggregator()
    customer_cat = cat_transformer.transform(raw)

    # 5. Merge all customer-level frames
    customer = (
        customer_agg
        .merge(customer_temporal, on="CustomerId", how="left")
        .merge(customer_rfm, on="CustomerId", how="left")
        .merge(customer_cat, on="CustomerId", how="left")
    )
    logger.info("Merged customer frame shape: %s", customer.shape)

    # 6. Proxy target (fit+transform on the merged frame)
    proxy_engineer = ProxyTargetEngineer(
        n_clusters=N_CLUSTERS, random_state=RANDOM_STATE
    )
    customer = proxy_engineer.fit_transform(customer)

    # 7. Drop zero-variance columns
    zero_var = ["CountryCode", "CurrencyCode"]  # confirmed constant in EDA
    customer = customer.drop(
        columns=[c for c in zero_var if c in customer.columns], errors="ignore"
    )

    # 8. Encode categoricals
    cat_cols = ["ProductCategory", "ChannelId", "ProviderId"]
    existing_cat = [c for c in cat_cols if c in customer.columns]

    if existing_cat:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore",
                            dtype=np.float32)
        encoded = ohe.fit_transform(customer[existing_cat])
        ohe_cols = ohe.get_feature_names_out(existing_cat)
        encoded_df = pd.DataFrame(
            encoded, columns=ohe_cols, index=customer.index
        )
        customer = pd.concat(
            [customer.drop(columns=existing_cat), encoded_df], axis=1
        )

    # 9. Convert PricingStrategy to numeric if still object
    if "PricingStrategy" in customer.columns:
        customer["PricingStrategy"] = pd.to_numeric(
            customer["PricingStrategy"], errors="coerce"
        )

    # 10. Impute residual missing values
    num_cols = customer.select_dtypes(include=[np.number]).columns.tolist()
    target_col = "is_high_risk"
    feature_num_cols = [c for c in num_cols if c != target_col]

    if feature_num_cols:
        imputer = SimpleImputer(strategy="median")
        customer[feature_num_cols] = imputer.fit_transform(
            customer[feature_num_cols]
        )

    # 11. Scale numerical features (exclude ID and target)
    scale_cols = [
        c for c in feature_num_cols
        if c not in ("CustomerId",)
    ]
    if scale_cols:
        scaler = StandardScaler()
        customer[scale_cols] = scaler.fit_transform(customer[scale_cols])

    logger.info("Final processed shape: %s", customer.shape)
    logger.info(
        "is_high_risk distribution:\n%s",
        customer["is_high_risk"].value_counts().to_string(),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    customer.to_csv(output_path, index=False)
    logger.info("Saved processed data to %s", output_path)

    return customer


# ── CLI entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full processing pipeline.")
    parser.add_argument("--input",  default="data/raw/data.csv",
                        help="Path to raw CSV file")
    parser.add_argument("--output", default="data/processed/processed.csv",
                        help="Path to write processed CSV file")
    args = parser.parse_args()
    run_pipeline(args.input, args.output)
