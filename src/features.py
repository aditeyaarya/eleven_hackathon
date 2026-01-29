from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st

from .data_utils import parse_date_series


def build_feature_form(cfg: dict) -> dict:
    """
    Build Streamlit inputs for all features required by the XGB pipeline.
    Returns dict {feature_name: value}.
    """
    num_features = cfg.get("num_features", [])
    cat_features = cfg.get("cat_features", [])

    st.subheader("Input features")
    values: dict = {}

    st.markdown("### Numeric")
    for f in num_features:
        if f.startswith("CntFam1_"):
            values[f] = int(st.number_input(f, min_value=0, value=0, step=1))
        else:
            values[f] = float(st.number_input(f, value=0.0))

    st.markdown("### Categorical")
    for f in cat_features:
        if f in ["ClientOptINEmail", "ClientOptINPhone"]:
            values[f] = st.selectbox(f, options=["0", "1"], index=0)
        else:
            values[f] = st.text_input(f, value="")

    return values


def build_client_features(client_id: str, tx: pd.DataFrame, cfg: dict) -> dict:
    """
    Rebuild the client-level feature vector.
    Uses:
      - RecencyDays based on most recent TransactionDate
      - Counts for top FamilyLevel1 values: CntFam1_<fam>
      - Opt-in fields if present: ClientOptINEmail/ClientOptINPhone
      - Any remaining categorical features: default ""
      - Any remaining numeric features: default 0
    """
    if "ClientID" not in tx.columns:
        raise ValueError("transactions.joblib must include a ClientID column.")

    client_tx = tx.loc[tx["ClientID"].astype(str) == str(client_id)].copy()
    if client_tx.empty:
        raise ValueError("No transactions found for this ClientID.")

    if "TransactionDate" not in client_tx.columns:
        raise ValueError("transactions.joblib must include a TransactionDate column.")

    client_tx["TransactionDate"] = parse_date_series(client_tx["TransactionDate"])
    last_date = client_tx["TransactionDate"].max()
    if pd.isna(last_date):
        raise ValueError("Could not parse TransactionDate for this client.")

    recency_days = (pd.Timestamp.now() - last_date).days

    feats: dict = {}
    feats["RecencyDays"] = int(recency_days)

    top_fam1 = cfg.get("top_fam1", [])
    if top_fam1:
        if "FamilyLevel1" not in client_tx.columns:
            raise ValueError(
                "transactions.joblib missing FamilyLevel1 needed for CntFam1_* features."
            )
        for fam in top_fam1:
            feats[f"CntFam1_{fam}"] = int(
                (client_tx["FamilyLevel1"].astype(str) == str(fam)).sum()
            )

    for opt in ["ClientOptINEmail", "ClientOptINPhone"]:
        if opt in client_tx.columns:
            v = client_tx[opt].dropna()
            feats[opt] = str(int(v.iloc[-1])) if len(v) else "0"
        else:
            feats[opt] = "0"

    num_features = cfg.get("num_features", [])
    cat_features = cfg.get("cat_features", [])

    for f in num_features:
        if f not in feats:
            feats[f] = 0

    for f in cat_features:
        if f not in feats:
            feats[f] = ""

    return feats
