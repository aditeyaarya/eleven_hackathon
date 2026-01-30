import pandas as pd
import numpy as np

def get_latest_feature_row_for_client(client_id, df_model):
    """
    Retrieves the latest feature row for a given client from the modeling dataframe.
    """
    cid = str(client_id)
    if "ClientID" not in df_model.columns:
         return None
         
    sub = df_model[df_model["ClientID"].astype(str) == cid].copy()
    if sub.empty:
        return None
        
    if "SaleTransactionDate" in sub.columns:
        sub = sub.sort_values("SaleTransactionDate")
        
    return sub.iloc[[-1]]

def predict_top_families_detailed(
    client_id, 
    df_model, 
    clf, 
    le, 
    products, 
    num_features, 
    cat_features, 
    top_n=5
):
    """
    Predicts top N FamilyLevel2 intents for a client and enriches with 
    Category, Universe, and FamilyLevel1 metadata.
    
    Parameters:
    - client_id: The client ID to predict for.
    - df_model: The dataframe containing client features (history).
    - clf: The trained XGBoost/Pipeline model.
    - le: The LabelEncoder for the target.
    - products: The products dataframe (for metadata lookup).
    - num_features: List of numerical feature names.
    - cat_features: List of categorical feature names.
    - top_n: Number of top predictions to return.
    
    Returns:
    - pd.DataFrame having columns: [FamilyLevel2, prob, Category, Universe, FamilyLevel1]
    """
    row = get_latest_feature_row_for_client(client_id, df_model)
    if row is None:
        return pd.DataFrame(columns=["FamilyLevel2", "prob", "Category", "Universe", "FamilyLevel1"])

    # Prepare input features
    # Ensure columns exist and order matches training
    cols = num_features + cat_features
    X_last = row[cols]
    
    # Predict
    p = clf.predict_proba(X_last)[0]
    
    # Get top N
    top_idx = np.argsort(p)[::-1][:top_n]
    fams = le.inverse_transform(top_idx)
    probs_ = p[top_idx]
    
    # Build initial dataframe
    res = pd.DataFrame({
        "FamilyLevel2": fams,
        "prob": probs_
    })
    
    # Lookup metadata from products
    # We create a lookup table unique by FamilyLevel2
    if "FamilyLevel2" in products.columns:
        lookup_cols = ["FamilyLevel2", "Category", "FamilyLevel1", "Universe"]
        # Filter to only cols that actually exist
        lookup_cols = [c for c in lookup_cols if c in products.columns]
        
        lookup = products[lookup_cols].drop_duplicates("FamilyLevel2")
        
        # Merge
        res = res.merge(lookup, on="FamilyLevel2", how="left").fillna("Unknown")
        
    return res
