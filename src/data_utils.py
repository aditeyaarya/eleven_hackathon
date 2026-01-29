import sys
from typing import Iterable, Optional, Set
import numpy as np
import pandas as pd


def patch_numpy_core_pickle_compat() -> None:
    """Pickle compat for numpy._core references."""
    try:
        import numpy.core as npcore
        sys.modules.setdefault("numpy._core", npcore)
    except Exception:
        pass


def parse_date_series(s: pd.Series) -> pd.Series:
    """
    Best-effort parse for TransactionDate columns.
    Always returns tz-naive datetimes (UTC normalized, tz removed),
    so arithmetic with pd.Timestamp.now() works reliably.
    """
    if np.issubdtype(s.dtype, np.datetime64):
        try:
            if getattr(s.dt, "tz", None) is not None:
                return s.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            pass
        return s

    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.dt.tz_convert("UTC").dt.tz_localize(None)


def safe_get_product_name_col(products_df: pd.DataFrame) -> str:
    """
    Pick the best available column that represents a human-readable product name.
    Falls back to ProductID only if nothing else exists.
    """
    candidates = [
        # common
        "ProductName", "Name", "Title", "Label", "Description",
        # retail / PIM / catalog variants
        "ProductLabel", "ProductTitle", "DisplayName", "ShortName", "LongName",
        "Model", "ModelName", "SKUName", "ItemName",
        "Product", "Product_Name", "product_name", "productTitle",
    ]
    for c in candidates:
        if c in products_df.columns:
            return c
    return "ProductID"


def get_in_stock_ids(
    stocks_df: Optional[pd.DataFrame],
    qty_col_candidates: Iterable[str] = ("Quantity", "Qty", "StockQty"),
) -> Set[str]:
    if stocks_df is None or stocks_df.empty:
        return set()

    if "ProductID" not in stocks_df.columns:
        return set()

    qty_col = None
    for c in qty_col_candidates:
        if c in stocks_df.columns:
            qty_col = c
            break

    if qty_col is None:
        return set(stocks_df["ProductID"].astype(str).unique())

    return set(
        stocks_df.loc[stocks_df[qty_col].fillna(0) > 0, "ProductID"]
        .astype(str)
        .unique()
    )


def build_reverse_index(product_index: dict) -> dict[int, str]:
    """product_index expected {ProductID -> idx}. Returns {idx -> ProductID}."""
    rev: dict[int, str] = {}
    for pid, idx in product_index.items():
        try:
            rev[int(idx)] = str(pid)
        except Exception:
            continue
    return rev
