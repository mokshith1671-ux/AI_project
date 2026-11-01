import os
import glob
import pandas as pd
import numpy as np

def load_and_preprocess_data(data_dir="data"):
    """
    Loads session datasets from data/, standardizes RP node names,
    filters persistent infrastructure APs, and returns train (Sessions 1-3)
    and test (Session 4) matrices.
    """
    session_files = {
        'Session_1': os.path.join(data_dir, 'Session_1_WiFi_Fingerprint_Readings_CORRECTED.xlsx'),
        'Session_2': os.path.join(data_dir, 'Session_2_WiFi_Fingerprint.csv'),
        'Session_3': os.path.join(data_dir, 'session3_raw_observations.csv'),
        'Session_4': os.path.join(data_dir, 'session4_raw_observations.csv')
    }

    all_data = []
    for s_name, path in session_files.items():
        if os.path.exists(path):
            df = pd.read_excel(path) if path.endswith('.xlsx') else pd.read_csv(path)
            df['session_id'] = s_name
            all_data.append(df)
        else:
            matches = glob.glob(os.path.join(data_dir, f"*{s_name.replace('_', '').lower()}*")) + \
                      glob.glob(os.path.join(data_dir, f"*{s_name}*"))
            if matches:
                m_path = matches[0]
                df = pd.read_excel(m_path) if m_path.endswith('.xlsx') else pd.read_csv(m_path)
                df['session_id'] = s_name
                all_data.append(df)

    if not all_data:
        raise FileNotFoundError("No session files found in the 'data/' directory.")

    # Combine sessions and reset index
    df_raw = pd.concat(all_data, ignore_index=True).reset_index(drop=True)

    # Standardize column headers and remove duplicate columns
    df_raw.columns = [str(col).strip().lower() for col in df_raw.columns]
    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()

    # Locate the RP / Location column
    loc_candidates = ['location_id', 'location', 'rp', 'reference_point']
    loc_col = next((c for c in loc_candidates if c in df_raw.columns), None)
    
    if not loc_col:
        obj_cols = [c for c in df_raw.select_dtypes(include=['object']).columns if c != 'session_id']
        if obj_cols:
            loc_col = obj_cols[0]
        else:
            raise KeyError("Could not locate a Location/RP column in the dataset files.")

    # Drop rows where location is null or missing
    df_raw = df_raw.dropna(subset=[loc_col]).reset_index(drop=True)

    # Safely derive Floor label: Ground Floor ('GF_') vs First Floor ('FF_')
    df_raw['floor'] = df_raw[loc_col].apply(
        lambda x: 'First Floor' if str(x).strip().upper().startswith('FF') else 'Ground Floor'
    )

    # Detect AP feature columns
    metadata_cols = [loc_col, 'session_id', 'floor', 'timestamp', 'ssid', 'bssid', 'rssi_dbm']
    ap_cols = [c for c in df_raw.columns if c not in metadata_cols]

    # Clean numeric AP values and fill missing signals with out-of-range sentinel value (-100 dBm)
    for col in ap_cols:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    df_raw[ap_cols] = df_raw[ap_cols].fillna(-100.0)

    # Filter top 19 persistent infrastructure APs across sessions
    if len(ap_cols) > 19:
        presence = (df_raw[ap_cols] > -100.0).sum(axis=0)
        top_19_aps = presence.nlargest(19).index.tolist()
    else:
        top_19_aps = ap_cols

    # Train (Sessions 1-3) vs Test (Session 4)
    test_mask = df_raw['session_id'] == 'Session_4'

    if not test_mask.any():
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df_raw, test_size=0.25, random_state=42, stratify=df_raw['floor'])
    else:
        train_df = df_raw[~test_mask].copy()
        test_df = df_raw[test_mask].copy()

    X_train = train_df[top_19_aps].values
    y_floor_train = train_df['floor'].values
    y_rp_train = train_df[loc_col].astype(str).str.strip().values

    X_test = test_df[top_19_aps].values
    y_floor_test = test_df['floor'].values
    y_rp_test = test_df[loc_col].astype(str).str.strip().values

    unique_rps = sorted(list(df_raw[loc_col].astype(str).str.strip().unique()))

    return X_train, y_floor_train, y_rp_train, X_test, y_floor_test, y_rp_test, top_19_aps, unique_rps