from __future__ import annotations

import numpy as np
import pandas as pd

from .data_utils import (
    safe_get_product_name_col,
    get_in_stock_ids,
    build_reverse_index,
    parse_date_series,
)
from .features import build_client_features


def predict_top_classes(xgb, le, feature_row: dict, top_n: int = 10) -> pd.DataFrame:
    X = pd.DataFrame([feature_row])
    proba = xgb.predict_proba(X)[0]
    classes = le.inverse_transform(np.arange(len(proba)))
    return (
        pd.DataFrame({"FamilyLevel2": classes, "prob": proba})
        .sort_values("prob", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def recommend_similar_products(
    products_df: pd.DataFrame,
    S,
    product_index: dict,
    seed_product_id: str,
    in_stock_ids: set,
    k: int = 10,
    familylevel2_filter: str | None = None,
) -> pd.DataFrame:
    pid = str(seed_product_id)

    if pid not in product_index:
        raise ValueError("Seed ProductID not found in product_index.")

    n = int(getattr(S, "shape", [len(S)])[0])
    seed_idx = int(product_index[pid])

    if seed_idx < 0 or seed_idx >= n:
        raise ValueError(
            f"Seed ProductID maps to index {seed_idx}, but similarity_matrix has only {n} rows. "
            "Your product_index and similarity_matrix are not aligned."
        )

    rev_index = build_reverse_index(product_index)
    sims = np.asarray(S[seed_idx]).ravel()
    order = np.argsort(-sims)

    rows = []
    for idx in order:
        idx = int(idx)
        if idx < 0 or idx >= n:
            continue
        if idx == seed_idx:
            continue

        cand_pid = rev_index.get(idx)
        if not cand_pid:
            continue

        if in_stock_ids and cand_pid not in in_stock_ids:
            continue

        cand_row = products_df.loc[products_df["ProductID"].astype(str) == cand_pid]
        if cand_row.empty:
            continue
        cand_row = cand_row.iloc[0].to_dict()

        if familylevel2_filter and "FamilyLevel2" in products_df.columns:
            if str(cand_row.get("FamilyLevel2", "")) != str(familylevel2_filter):
                continue

        out = {"ProductID": cand_pid, "similarity": float(sims[idx])}
        for c in ["ProductName", "Category", "FamilyLevel1", "Universe", "FamilyLevel2"]:
            if c in products_df.columns:
                out[c] = cand_row.get(c, "")
        rows.append(out)

        if len(rows) >= k:
            break

    return pd.DataFrame(rows)


def recommend_for_client(
    client_id: str,
    bundle: dict,
    top_families: int = 3,
    top_products: int = 10,
    per_family_candidates: int = 50,
) -> dict:
    tx = bundle["transactions"].copy()

    allowed = bundle.get("allowed_client_ids")
    if allowed is not None and str(client_id) not in set(allowed):
        raise ValueError("ClientID not in predefined clients.csv whitelist.")

    products = bundle["products"].copy()
    stocks = bundle["stocks"]
    xgb = bundle["xgb"]
    le = bundle["le"]
    cfg = bundle["cfg"]
    S = bundle["S"]
    product_index = bundle["product_index"]

    in_stock_ids = get_in_stock_ids(stocks)

    if "ProductID" not in products.columns:
        raise ValueError("products.joblib must include ProductID.")
    products["ProductID"] = products["ProductID"].astype(str)

    if "ProductID" in tx.columns:
        tx["ProductID"] = tx["ProductID"].astype(str)

    # --- similarity/index sizes & basic coverage ---
    nS = int(getattr(S, "shape", [len(S)])[0])
    nIndex = len(product_index) if isinstance(product_index, dict) else 0

    # --- client tx history ---
    client_tx = tx.loc[tx["ClientID"].astype(str) == str(client_id)].copy()
    if "TransactionDate" in client_tx.columns:
        client_tx["TransactionDate"] = parse_date_series(client_tx["TransactionDate"])
        client_tx = client_tx.sort_values("TransactionDate", ascending=False)

    history_cols = [
        c for c in ["TransactionDate", "ProductID", "FamilyLevel1", "FamilyLevel2"]
        if c in client_tx.columns
    ]
    client_last_purchases = client_tx[history_cols].head(10) if history_cols else client_tx.head(10)

    # --- seed logic: keep BOTH "raw" (latest purchase) and "usable" (in index + in bounds) ---
    seed_pid_raw = None
    if "ProductID" in client_tx.columns and not client_tx.empty:
        s = client_tx["ProductID"].dropna().astype(str).tolist()
        seed_pid_raw = s[0] if len(s) else None

    seed_pid_used = None
    seed_idx_used = None
    if "ProductID" in client_tx.columns and not client_tx.empty:
        for pid in client_tx["ProductID"].dropna().astype(str).tolist():
            if pid in product_index:
                idx = int(product_index[pid])
                if 0 <= idx < nS:
                    seed_pid_used = pid
                    seed_idx_used = idx
                    break

    # --- coverage diagnostics: how many bought products exist in product_index ---
    tx_pids = []
    if "ProductID" in client_tx.columns and not client_tx.empty:
        tx_pids = client_tx["ProductID"].dropna().astype(str).unique().tolist()

    in_index = [p for p in tx_pids if p in product_index]
    in_index_in_bounds = []
    for p in in_index:
        try:
            idx = int(product_index[p])
            if 0 <= idx < nS:
                in_index_in_bounds.append(p)
        except Exception:
            continue

    # --- features + family probabilities ---
    feats = build_client_features(client_id, tx, cfg)
    fam_probs = predict_top_classes(xgb, le, feats, top_n=top_families)

    # --- build recs + diagnostics per family ---
    rec_rows: list[dict] = []
    fam_diag_rows: list[dict] = []

    sims = None
    if seed_idx_used is not None:
        sims = np.asarray(S[seed_idx_used]).ravel()

    name_col = safe_get_product_name_col(products)

    # ✅ DEDUPE: by product NAME (preferred); if name missing, fallback to FamilyLevel2
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()

    def _norm(s: object) -> str:
        return str(s or "").strip().lower()

    def _dedupe_key(row: pd.Series, pid: str, fam2: str) -> str:
        """
        Primary: normalized product name (name_col).
        Fallback: FamilyLevel2 (so the same 'Trek Domane SL7' doesn't show 3 times
                  when ProductName is missing / empty).
        Final fallback: pid (should rarely be used).
        """
        nm = _norm(row.get(name_col, ""))
        if nm and nm != "nan" and name_col != "ProductID":
            return f"name:{nm}"

        # if we do not have a real name column, dedupe by intent label instead
        fam2n = _norm(fam2)
        if fam2n and fam2n != "nan":
            return f"fam2:{fam2n}"

        return f"pid:{pid}"

    for _, fr in fam_probs.iterrows():
        fam2 = str(fr["FamilyLevel2"])
        fam_prob = float(fr["prob"])

        if "FamilyLevel2" in products.columns:
            fam_pool_all = products.loc[(products["FamilyLevel2"].astype(str) == fam2)].copy()
        else:
            fam_pool_all = products.copy()

        cnt_total = int(len(fam_pool_all))

        fam_pool = fam_pool_all
        if in_stock_ids:
            fam_pool = fam_pool.loc[fam_pool["ProductID"].isin(in_stock_ids)]
        cnt_in_stock = int(len(fam_pool))

        cnt_in_index_bounds = None

        if sims is not None:
            def _idx_or_neg1(x: str) -> int:
                if x in product_index:
                    try:
                        idx = int(product_index[x])
                        if 0 <= idx < nS:
                            return idx
                    except Exception:
                        pass
                return -1

            fam_pool = fam_pool.copy()
            fam_pool["__idx__"] = fam_pool["ProductID"].map(_idx_or_neg1)
            fam_pool = fam_pool.loc[fam_pool["__idx__"] >= 0].copy()
            cnt_in_index_bounds = int(len(fam_pool))

            if not fam_pool.empty:
                fam_pool["sim_to_last"] = fam_pool["__idx__"].map(lambda i: float(sims[int(i)]))
                fam_pool = fam_pool.sort_values("sim_to_last", ascending=False).head(per_family_candidates)

                for _, p in fam_pool.iterrows():
                    pid = str(p["ProductID"])
                    key = _dedupe_key(p, pid, fam2)

                    if pid in seen_ids or key in seen_keys:
                        continue

                    seen_ids.add(pid)
                    seen_keys.add(key)

                    rec_rows.append({
                        "ClientID": str(client_id),
                        "SeedProductID": str(seed_pid_raw or ""),
                        "FamilyLevel2": fam2,
                        "FamilyProb": fam_prob,
                        "ProductID": pid,
                        "SimilarityToLastPurchase": float(p.get("sim_to_last", 0.0)),
                        name_col: p.get(name_col, ""),
                    })

        else:
            if not fam_pool.empty:
                fam_pool = fam_pool.head(per_family_candidates)

                for _, p in fam_pool.iterrows():
                    pid = str(p["ProductID"])
                    key = _dedupe_key(p, pid, fam2)

                    if pid in seen_ids or key in seen_keys:
                        continue

                    seen_ids.add(pid)
                    seen_keys.add(key)

                    rec_rows.append({
                        "ClientID": str(client_id),
                        "SeedProductID": str(seed_pid_raw or ""),
                        "FamilyLevel2": fam2,
                        "FamilyProb": fam_prob,
                        "ProductID": pid,
                        "SimilarityToLastPurchase": np.nan,
                        name_col: p.get(name_col, ""),
                    })

        fam_diag_rows.append({
            "FamilyLevel2": fam2,
            "prob": fam_prob,
            "products_total_in_family": cnt_total,
            "products_in_stock_in_family": cnt_in_stock,
            "products_in_index_and_in_bounds": cnt_in_index_bounds if cnt_in_index_bounds is not None else "N/A (no seed)",
        })

    if not rec_rows:
        recs = pd.DataFrame(columns=[
            "ClientID", "SeedProductID", "FamilyLevel2", "FamilyProb",
            "ProductID", "SimilarityToLastPurchase", name_col
        ])
    else:
        recs = pd.DataFrame(rec_rows)
        recs["SimilarityToLastPurchase"] = pd.to_numeric(recs["SimilarityToLastPurchase"], errors="coerce")
        recs = recs.sort_values(
            ["FamilyProb", "SimilarityToLastPurchase"],
            ascending=[False, False],
            na_position="last",
        )

        # extra safety net: dedupe again using the same key logic
        if not recs.empty:
            nm = recs.get(name_col, pd.Series([""] * len(recs))).astype(str).str.strip().str.lower()
            if name_col != "ProductID":
                key_series = np.where((nm != "") & (nm != "nan"), "name:" + nm, "fam2:" + recs["FamilyLevel2"].astype(str).str.lower())
            else:
                key_series = "fam2:" + recs["FamilyLevel2"].astype(str).str.lower()

            recs["__key__"] = key_series
            recs = recs.drop_duplicates(subset=["__key__"]).drop(columns=["__key__"])
            recs = recs.drop_duplicates(subset=["ProductID"]).head(top_products).reset_index(drop=True)

    diagnostics = {
        "client_id": str(client_id),
        "client_transactions": int(len(client_tx)),
        "client_unique_products": int(len(set(tx_pids))),
        "seed_pid_raw_latest_purchase": str(seed_pid_raw) if seed_pid_raw else None,
        "seed_pid_used_for_similarity": str(seed_pid_used) if seed_pid_used else None,
        "seed_idx_used_for_similarity": int(seed_idx_used) if seed_idx_used is not None else None,
        "similarity_matrix_rows": int(nS),
        "product_index_size": int(nIndex),
        "client_bought_products_in_product_index": int(len(in_index)),
        "client_bought_products_in_index_and_in_bounds": int(len(in_index_in_bounds)),
        "predicted_families": fam_probs.to_dict(orient="records"),
        "family_candidate_counts": fam_diag_rows,
        "final_recommendations_returned": int(len(recs)),
        "requested_top_products": int(top_products),
        "name_col_used_for_display": str(name_col),
        "note": (
            "If ProductName is missing/empty, we dedupe by FamilyLevel2 (Intent) to avoid showing the same named product multiple times."
        )
    }

    return {
        "fam_probs": fam_probs,
        "recs": recs,
        "client_last_purchases": client_last_purchases.reset_index(drop=True),
        "features_used": pd.DataFrame([feats]),
        "diagnostics": diagnostics,
    }
