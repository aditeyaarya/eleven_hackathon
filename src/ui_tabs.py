from __future__ import annotations

import re
import numpy as np
import pandas as pd
import streamlit as st

from .data_utils import safe_get_product_name_col, get_in_stock_ids
from .features import build_feature_form
from .recommenders import (
    predict_top_classes,
    recommend_for_client,
    recommend_similar_products,
)

# -----------------------------
# Business-friendly helpers
# -----------------------------




def _confidence_label(p: float) -> str:
    if p >= 0.35:
        return "High"
    if p >= 0.18:
        return "Medium"
    return "Low"


def _match_label(sim: float | None) -> str:
    if sim is None or (isinstance(sim, float) and np.isnan(sim)):
        return "N/A"
    if sim >= 0.65:
        return "Very high"
    if sim >= 0.45:
        return "High"
    if sim >= 0.25:
        return "Medium"
    return "Low"


def _build_reason(seed_pid: str, fam2: str, fam_prob: float, sim: float | None) -> str:
    """
    IMPORTANT: You asked to keep this unchanged.
    It still includes '(Low/Medium/High confidence)' in its returned string.
    """
    parts = []
    if seed_pid:
        parts.append("Based on customer’s recent purchase history")
    parts.append(f"Predicted intent: **{fam2}** ({_confidence_label(fam_prob)} confidence)")
    if sim is not None and not (isinstance(sim, float) and np.isnan(sim)):
        parts.append(f"Product match to last purchase: **{_match_label(float(sim))}**")
    return " • ".join(parts)


def _strip_confidence_from_reason(reason: str) -> str:
    """
    UI-only sanitization:
    Removes '(... confidence)' from the displayed reason line
    WITHOUT changing _build_reason() output.
    """
    # Removes: " (Low confidence)" / " (Medium confidence)" / " (High confidence)" (case-insensitive)
    return re.sub(r"\s*\([^)]*confidence\)", "", reason, flags=re.IGNORECASE).strip()


def _render_recommendation_cards(
    recs: pd.DataFrame,
    name_col: str,
    show_similarity: bool,
) -> None:
    if recs is None or recs.empty:
        st.warning("No recommendations to display (after filters).")
        return

    for i, row in recs.reset_index(drop=True).iterrows():
        rank = i + 1
        # Requested change: Display Product Name (FamilyLevel2) instead of ProductID/Name
        title = str(row.get("FamilyLevel2", ""))
        
        # New info for description
        cat = str(row.get("Category", ""))
        uni = str(row.get("Universe", ""))
        fam1 = str(row.get("FamilyLevel1", ""))
        
        # Formatting description
        desc_parts = []
        if cat: desc_parts.append(f"Category: **{cat}**")
        if uni: desc_parts.append(f"Universe: **{uni}**")
        if fam1: desc_parts.append(f"Family: **{fam1}**")
        description = " • ".join(desc_parts)

        with st.container(border=True):
            # Use full width since we are removing the Intent column
            st.markdown(f"### #{rank} — {title}")
            if description:
                st.markdown(description)

# -----------------------------
# Tabs
# -----------------------------

def render_tab_client_recommender(bundle: dict) -> None:
    """
    BUSINESS UI:
    - Strategy preset instead of ML knobs
    - Filters that map to business decisions
    - Card-style ranked recommendations
    - Export actions
    - Optional "Explain/Diagnostics" expanders
    """
    st.subheader("Customer Recommendations")

    allowed = bundle.get("allowed_client_ids") or []
    products = bundle["products"]
    stocks = bundle["stocks"]

    in_stock_ids = get_in_stock_ids(stocks)
    name_col = safe_get_product_name_col(products)

    # ---------- Controls ----------
    top = st.container(border=True)
    with top:
        c1, c2, c3 = st.columns([2.4, 1.6, 1.0])

        with c1:
            if allowed:
                client_id = st.selectbox(
                    "Customer (ClientID)",
                    options=allowed,
                    index=0,
                    help="Select a customer from your allowed client list (clients.csv).",
                    key="client_recommender_client_id",
                )
            else:
                client_id = st.text_input("Customer (ClientID)", value="", key="client_recommender_client_id_manual")
                st.caption("clients.csv whitelist not loaded → using manual ClientID input.")



        with c3:
            top_products = st.number_input(
                "How many products?",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                key="client_recommender_top_products",
            )

        f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1.2])
        with f1:
            in_stock_only = st.checkbox(
                "In-stock only",
                value=True,
                key="client_recommender_in_stock_only",
            )
        with f2:
            exclude_already_bought = st.checkbox(
                "Exclude already bought",
                value=True,
                key="client_recommender_exclude_already_bought",
            )
        with f3:
            diversify = st.checkbox(
                "Diversify results",
                value=True,
                help="Keeps variety across families but still returns the requested count (fills remainder by rank).",
                key="client_recommender_diversify",
            )
        with f4:
            show_explain = st.checkbox(
                "Show explanations",
                value=True,
                key="client_recommender_show_explain",
            )

        run = st.button("Generate recommendations", type="primary", key="client_recommender_run")

    if not run:
        st.info("Select a customer and click **Generate recommendations**.")
        return

    # ---------- Map strategy → parameters ----------
    # Hardcoded "Balanced" defaults as requested
    top_families = 10
    per_family_candidates = 80

    # ---------- Compute ----------
    try:
        out = recommend_for_client(
            client_id=str(client_id).strip(),
            bundle=bundle,
            top_families=top_families,
            top_products=int(top_products),
            per_family_candidates=per_family_candidates,
        )
    except Exception as e:
        st.error(f"Failed: {e}")
        return

    recs = out.get("recs", pd.DataFrame()).copy()
    history = out.get("client_last_purchases", pd.DataFrame()).copy()
    fam_probs_df = out.get("fam_probs", pd.DataFrame()).copy()
    features_used_df = out.get("features_used", pd.DataFrame()).copy()
    diagnostics = out.get("diagnostics", {})

    # ---------- Filters ----------
    if in_stock_only and in_stock_ids and not recs.empty and "ProductID" in recs.columns:
        recs = recs.loc[recs["ProductID"].astype(str).isin(in_stock_ids)].copy()

    if (
        exclude_already_bought
        and not history.empty
        and "ProductID" in history.columns
        and not recs.empty
        and "ProductID" in recs.columns
    ):
        bought = set(history["ProductID"].dropna().astype(str).unique())
        recs = recs.loc[~recs["ProductID"].astype(str).isin(bought)].copy()

    # ---------- Diversify but ALWAYS return exactly top_products (if possible) ----------
    if diversify and not recs.empty:
        sort_cols = ["FamilyProb"]
        if "SimilarityToLastPurchase" in recs.columns:
            sort_cols.append("SimilarityToLastPurchase")

        recs = recs.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last").copy()

        if "FamilyLevel2" not in recs.columns:
            recs = recs.head(int(top_products)).reset_index(drop=True)
        else:
            cap_per_family = 2

            recs["__rank_in_family__"] = recs.groupby("FamilyLevel2").cumcount()
            diversified = recs.loc[recs["__rank_in_family__"] < cap_per_family].copy()

            if len(diversified) < int(top_products):
                remaining = recs.loc[recs["__rank_in_family__"] >= cap_per_family].copy()
                need = int(top_products) - len(diversified)
                fill = remaining.head(need)
                diversified = pd.concat([diversified, fill], ignore_index=True)

            recs = diversified.drop(columns=["__rank_in_family__"]).head(int(top_products)).reset_index(drop=True)
    else:
        if not recs.empty:
            recs = recs.head(int(top_products)).reset_index(drop=True)

    if len(recs) < int(top_products):
        st.warning(
            f"Only {len(recs)} products available after filters (requested {int(top_products)}). "
            "Try turning off 'Exclude already bought' or 'In-stock only'."
        )

    # ---------- Summary ----------
    s1, s2, s3, s4 = st.columns(4)

    with s2:
        st.metric("Products shown", len(recs))
    with s3:
        st.metric("In-stock filter", "On" if in_stock_only else "Off")
    with s4:
        seed_pid = ""
        if not recs.empty and "SeedProductID" in recs.columns:
            seed_pid = str(recs["SeedProductID"].iloc[0])
        st.metric("Seed product", seed_pid if seed_pid else "N/A")

    # ---------- Cards ----------
    st.markdown("## Recommended Products")
    _render_recommendation_cards(
        recs=recs,
        name_col=name_col,
        show_similarity=show_explain,
    )

    # ---------- Export ----------
    st.divider()
    cexp1, cexp2 = st.columns([1, 1])
    with cexp1:
        csv_bytes = recs.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download recommendations (CSV)",
            data=csv_bytes,
            file_name=f"recs_client_{client_id}.csv",
            mime="text/csv",
            key=f"download_recs_{client_id}",
        )
    with cexp2:
        st.caption("Results.")

    # ---------- Explainability / diagnostics ----------
    if show_explain:
        with st.expander("Why these recommendations? (Model signals)", expanded=False):
            st.markdown("### Predicted next intent (FamilyLevel2)")
            st.dataframe(fam_probs_df, use_container_width=True)

        with st.expander("Customer purchase context (last 10 purchases)", expanded=False):
            st.dataframe(history, use_container_width=True)

        with st.expander("Technical view (features used by the model)", expanded=False):
            st.dataframe(features_used_df, use_container_width=True)

        with st.expander("Diagnostics (why only a few products / why Seed or Match is N/A)", expanded=False):
            st.json(diagnostics)


def render_tab_manual_prediction(bundle: dict) -> None:
    """
    DS/Advanced tab: manual feature entry → top FamilyLevel2 predictions
    """
    st.subheader("Manual Prediction (Advanced)")

    cfg = bundle["cfg"]
    xgb = bundle["xgb"]
    le = bundle["le"]

    feature_row = build_feature_form(cfg)

    top_n = st.slider("Show top N families", min_value=3, max_value=20, value=10, key="manual_top_n")
    if st.button("Predict (manual input)", key="manual_predict_btn"):
        try:
            df = predict_top_classes(xgb, le, feature_row, top_n=int(top_n))
            st.success("Prediction generated.")
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Failed: {e}")


def render_tab_similar_products(bundle: dict) -> None:
    """
    Business tab: find substitutes/alternatives for a product (in-stock)
    """
    st.subheader("Similar Products (for substitution / merchandising)")

    products = bundle["products"]
    stocks = bundle["stocks"]
    S = bundle["S"]
    product_index = bundle["product_index"]

    in_stock_ids = get_in_stock_ids(stocks)
    name_col = safe_get_product_name_col(products)

    c1, c2, c3 = st.columns([2.0, 1.0, 1.0])

    with c1:
        seed_product_id = st.text_input("Seed ProductID", value="", key="similar_seed_product_id")
    with c2:
        k = st.number_input("How many?", min_value=1, max_value=50, value=10, step=1, key="similar_k")
    with c3:
        in_stock_only = st.checkbox(
            "In-stock only",
            value=True,
            key="similar_products_in_stock_only",
        )

    fam2_filter = st.text_input("Optional: filter to FamilyLevel2", value="", key="similar_fam2_filter")

    if st.button("Find similar products", type="primary", key="similar_find_btn"):
        try:
            ids = in_stock_ids if in_stock_only else set()
            df = recommend_similar_products(
                products_df=products,
                S=S,
                product_index=product_index,
                seed_product_id=str(seed_product_id).strip(),
                in_stock_ids=ids,
                k=int(k),
                familylevel2_filter=fam2_filter.strip() or None,
            )
            if df.empty:
                st.warning("No similar products found (check ProductID / stock / filters).")
                return

            cols = [
                c
                for c in [
                    "ProductID",
                    name_col,
                    "Category",
                    "FamilyLevel1",
                    "Universe",
                    "FamilyLevel2",
                    "similarity",
                ]
                if c in df.columns
            ]
            st.dataframe(df[cols], use_container_width=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download similar products (CSV)",
                data=csv_bytes,
                file_name=f"similar_products_{seed_product_id}.csv",
                mime="text/csv",
                key=f"download_similar_{seed_product_id}",
            )
        except Exception as e:
            st.error(f"Failed: {e}")
