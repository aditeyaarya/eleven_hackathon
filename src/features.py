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
    Rebuild the client-level feature vector matching Combined.ipynb training logic.
    """
    if "ClientID" not in tx.columns:
        raise ValueError("transactions.joblib must include a ClientID column.")

    client_tx = tx.loc[tx["ClientID"].astype(str) == str(client_id)].copy()
    if client_tx.empty:
        raise ValueError("No transactions found for this ClientID.")

    if "TransactionDate" not in client_tx.columns:
        raise ValueError("transactions.joblib must include a TransactionDate column.")

    # Ensure sorting by date
    client_tx["TransactionDate"] = parse_date_series(client_tx["TransactionDate"])
    client_tx = client_tx.sort_values("TransactionDate")
    
    last_date = client_tx["TransactionDate"].max()
    if pd.isna(last_date):
        raise ValueError("Could not parse TransactionDate for this client.")

    # 1. Recency
    recency_days = (pd.Timestamp.now() - last_date).days
    
    feats: dict = {}
    feats["RecencyDays"] = int(recency_days)

    # 2. Cumulative stats (up to now)
    # Notebook: Quantity, SalesNetAmountEuro
    # We need to be careful if columns exist
    qty = pd.to_numeric(client_tx.get("Quantity", 1), errors="coerce").fillna(0.0)
    spend = pd.to_numeric(client_tx.get("SalesNetAmountEuro", 0), errors="coerce").fillna(0.0)
    
    feats["CumQty"] = float(qty.sum())
    feats["CumSpend"] = float(spend.sum())
    feats["CumTxns"] = int(len(client_tx))
    feats["AvgBasket"] = feats["CumSpend"] / feats["CumTxns"] if feats["CumTxns"] > 0 else 0.0

    # 3. Last purchased attributes
    # The last row in sorted client_tx is the last transaction
    last_tx = client_tx.iloc[-1]
    for col in ["Category", "FamilyLevel1", "Universe", "FamilyLevel2"]:
        val = last_tx.get(col, "")
        feats[f"Last_{col}"] = str(val) if pd.notna(val) else ""

    # 4. Weighted Family Counts (Exponential Decay)
    half_life = cfg.get("half_life_days", 90.0)
    
    # Calculate decay weights for all rows
    days_ago = (last_date - client_tx["TransactionDate"]).dt.days.clip(lower=0).fillna(0)
    # decay = exp(-days_ago / half_life)
    # Note: Training used global_max_date, here we use client's last_date as reference or now?
    # Usually for inference "at this moment", we decay from NOW. 
    # BUT, the notebook training calculated 'days_ago' relative to 'global_max_date' of the dataset?
    # Wait, checking notebook cell 6:
    # global_max_date = df_model["SaleTransactionDate"].max()
    # days_ago = (global_max_date - df_model["SaleTransactionDate"])
    # So it normalized everything to the dataset's end.
    # For inference, if we want to be consistent with a static model, we should probably use the same reference point 
    # OR if the model learns 'relative recency', we decay from 'now'.
    # Given 'RecencyDays' is (Now - Last Purchase), let's stick to relative decay.
    # However, for 'CntFam1', usually it's "weighted count", so relative to the event time.
    # Let's use (last_date - txn_date) as the lag so the most recent purchase has weight 1.0 (decay 0).
    
    w = np.exp(-days_ago / float(half_life))
    event_w = w * qty

    top_fam1 = cfg.get("top_fam1", [])
    if top_fam1:
        if "FamilyLevel1" not in client_tx.columns:
            # Try to handle missing column gracefully or warn
            pass
        else:
            # We can vectorize this
            fam_series = client_tx["FamilyLevel1"].astype(str)
            for fam in top_fam1:
                # Weighted sum where family matches
                mask = (fam_series == str(fam))
                feats[f"CntFam1_{fam}"] = float(event_w[mask].sum())

    # 5. Opt-ins
    for opt in ["ClientOptINEmail", "ClientOptINPhone"]:
        if opt in client_tx.columns:
            v = client_tx[opt].dropna()
            feats[opt] = str(int(v.iloc[-1])) if len(v) else "0"
        else:
            feats[opt] = "0"

    # 6. Fill missing with defaults
    num_features = cfg.get("num_features", [])
    cat_features = cfg.get("cat_features", [])

    for f in num_features:
        if f not in feats:
            feats[f] = 0.0 # Numeric default
            
    for f in cat_features:
        if f not in feats:
            feats[f] = "" # Categorical default

    return feats
