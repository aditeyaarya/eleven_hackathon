import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

from .data_utils import patch_numpy_core_pickle_compat


def _load_df_from_csv_or_joblib(bundle: Path, csv_rel: str, joblib_rel: str) -> pd.DataFrame:
    csv_path = bundle / csv_rel
    jl_path = bundle / joblib_rel

    if csv_path.exists():
        return pd.read_csv(csv_path)

    obj = joblib.load(jl_path)
    if isinstance(obj, dict):
        return pd.DataFrame(obj)
    if isinstance(obj, pd.DataFrame):
        return obj
    return pd.DataFrame(obj)


def _normalize_transactions_date_column(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions is None or transactions.empty:
        return transactions

    if "TransactionDate" in transactions.columns:
        return transactions

    candidates = [
        "SaleTransactionDate",  # your column
        "SaleDate",
        "Transaction_Date",
        "TxnDate",
        "Date",
        "Datetime",
        "Timestamp",
    ]
    for c in candidates:
        if c in transactions.columns:
            transactions = transactions.rename(columns={c: "TransactionDate"})
            break

    return transactions


def _enrich_transactions_with_products(transactions: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """
    Add Category/FamilyLevel1/Universe/FamilyLevel2 into transactions via ProductID join.
    This is required for CntFam1_* features and last purchase context.
    """
    if transactions is None or transactions.empty:
        return transactions
    if products is None or products.empty:
        return transactions

    if "ProductID" not in transactions.columns or "ProductID" not in products.columns:
        return transactions

    # Ensure string join keys
    tx = transactions.copy()
    tx["ProductID"] = tx["ProductID"].astype(str)

    prod_cols = [c for c in ["ProductID", "Category", "FamilyLevel1", "Universe", "FamilyLevel2"] if c in products.columns]
    if prod_cols == ["ProductID"]:
        return tx

    prod = products[prod_cols].copy()
    prod["ProductID"] = prod["ProductID"].astype(str)

    # Avoid duplicate cols if already present in tx
    to_add = [c for c in prod_cols if c != "ProductID" and c not in tx.columns]
    if not to_add:
        return tx

    prod = prod[["ProductID"] + to_add]

    # Left join: keep all transactions, enrich where possible
    tx = tx.merge(prod, on="ProductID", how="left")
    return tx


@st.cache_resource(show_spinner=True)
def load_bundle(bundle_dir: str) -> dict:
    patch_numpy_core_pickle_compat()
    bundle = Path(bundle_dir)

    # ---- Artifacts / models ----
    with open(bundle / "st/artifacts.json", "r") as f:
        cfg = json.load(f)

    xgb_pipeline = joblib.load(bundle / "st/xgb_pipeline.joblib")
    label_encoder = joblib.load(bundle / "st/label_encoder.joblib")
    product_index = joblib.load(bundle / "st/product_index.joblib")
    similarity_matrix = joblib.load(bundle / "st/similarity_matrix.joblib", mmap_mode="r")

    # ---- Data (prefer CSV under data/) ----
    products = _load_df_from_csv_or_joblib(bundle, "data/products.csv", "st/products.joblib")
    stocks = _load_df_from_csv_or_joblib(bundle, "data/stocks.csv", "st/stocks.joblib")
    transactions = _load_df_from_csv_or_joblib(bundle, "data/transactions.csv", "st/transactions.joblib")

    # ✅ Fix 1: normalize date column name
    transactions = _normalize_transactions_date_column(transactions)

    # ✅ Fix 2: bring FamilyLevel1/etc into transactions
    transactions = _enrich_transactions_with_products(transactions, products)

    clients_path = bundle / "data/clients.csv"
    clients = pd.read_csv(clients_path) if clients_path.exists() else None

    # ---- Allowed clients whitelist ----
    allowed_client_ids = None
    if clients is not None and not clients.empty:
        cid_col = None
        for c in ["ClientID", "client_id", "clientID", "CLIENTID", "IdClient", "IDClient"]:
            if c in clients.columns:
                cid_col = c
                break
        if cid_col is None:
            raise ValueError("clients.csv must contain a ClientID column (e.g., ClientID).")

        allowed_client_ids = clients[cid_col].dropna().astype(str).unique().tolist()

        if isinstance(transactions, pd.DataFrame) and "ClientID" in transactions.columns:
            present = set(transactions["ClientID"].astype(str).unique())
            allowed_client_ids = [cid for cid in allowed_client_ids if cid in present]

    return {
        "cfg": cfg,
        "xgb": xgb_pipeline,
        "le": label_encoder,
        "products": products,
        "stocks": stocks,
        "transactions": transactions,
        "clients": clients,
        "allowed_client_ids": allowed_client_ids,
        "product_index": product_index,
        "S": similarity_matrix,
    }
