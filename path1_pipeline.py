import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# STEP 1: DATA EXTRACTION & PREPROCESSING
# ==========================================
def load_and_clean_data(file_path):
    print("[1/4] Loading and cleaning raw Wi-Fi observations...")
    df_raw = pd.read_excel(file_path, sheet_name='All AP Observations')
    
    # Drop rows missing critical location or scan numbers
    df_clean = df_raw.dropna(subset=['Location', 'Scan No.']).copy()
    
    # Standardize AP identifier using BSSID MAC or SSID fallback
    df_clean['AP_ID'] = df_clean['BSSID / MAC'].fillna(df_clean['SSID / Access Point'])
    
    # Map and standardize location names to consistent physical nodes
    loc_mapping = {
        'Ground Floor - Library Entrance': 'GF_Library_Entrance',
        'Ground Floor Library Entrance': 'GF_Library_Entrance',
        'Ground Floor - Study Hall Entrance': 'GF_StudyHall_Entrance',
        'Ground Floor Study Hall Entrance': 'GF_StudyHall_Entrance',
        'Ground Floor Study Hall Centre': 'GF_StudyHall_Centre',
        'Library Hall 2 Entrance': 'GF_LibraryHall2_Entrance',
        'Library Hall 2 Centre': 'GF_LibraryHall2_Centre',
        'Staircase Ground Floor End': 'Staircase_GF',
        'Staircase First Floor End': 'Staircase_FF',
        'First-floor staircase end': 'Staircase_FF',
        'Computer Entrance': 'FF_ComputerCentre_Entrance',
        'First Floor – Computer Centre Entrance': 'FF_ComputerCentre_Entrance',
        'First-floor Computer Centre entrance': 'FF_ComputerCentre_Entrance',
        "Ojah Ma'am's cabin entrance (First Floor)": 'FF_OjahCabin_Entrance',
        'First Floor Balcony': 'FF_Balcony'
    }
    
    df_clean['Clean_Location'] = df_clean['Location'].map(loc_mapping)
    df_clean = df_clean.dropna(subset=['Clean_Location'])
    
    # Pivot long format (AP lines) into wide matrix (Fingerprint samples)
    pivot_df = df_clean.pivot_table(
        index=['Clean_Location', 'Scan No.'], 
        columns='AP_ID', 
        values='Signal (dBm)', 
        aggfunc='mean'
    ).fillna(-100) # Fill unobserved APs with missing signal sentinel -100 dBm
    
    dataset = pivot_df.reset_index()
    print(f"Dataset successfully built: {dataset.shape[0]} total fingerprints across {dataset['Clean_Location'].nunique()} locations with {pivot_df.shape[1]} AP features.\n")
    return dataset

# ==========================================
# STEP 2: ML & DL MODEL COMPARISON
# ==========================================
def evaluate_models(dataset):
    print("[2/4] Training and evaluating Machine Learning & Deep Learning models...")
    
    feature_cols = [c for c in dataset.columns if c not in ['Clean_Location', 'Scan No.']]
    X = dataset[feature_cols].values
    y_raw = dataset['Clean_Location'].values
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Define models
    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost Classifier": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='mlogloss'),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    }
    
    # Stratified K-Fold Evaluation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    print("--- 3-Fold Stratified Cross-Validation Results ---")
    for name, model in models.items():
        scoring = ['accuracy', 'f1_macro']
        cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring)
        mean_acc = cv_results['test_accuracy'].mean() * 100
        mean_f1 = cv_results['test_f1_macro'].mean()
        print(f"{name:25s} | Accuracy: {mean_acc:6.2f}% | Macro F1: {mean_f1:.4f}")
    
    # Train test split for detailed metrics report
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    selected_model = RandomForestClassifier(n_estimators=100, random_state=42)
    selected_model.fit(X_train, y_train)
    y_pred = selected_model.predict(X_test)
    
    print("\n--- Detailed Holdout Evaluation (Random Forest) ---")
    print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    return selected_model, le, feature_cols

# ==========================================
# STEP 3: INDOOR MAP GRAPH & PATHFINDING
# ==========================================
def build_building_graph():
    print("\n[3/4] Initializing building floorplan topological graph...")
    G = nx.Graph()
    
    # Edges represent walkable corridors/stairs with distance weights in meters
    edges = [
        ('GF_Library_Entrance', 'GF_StudyHall_Entrance', 10),
        ('GF_StudyHall_Entrance', 'GF_StudyHall_Centre', 12),
        ('GF_Library_Entrance', 'GF_LibraryHall2_Entrance', 8),
        ('GF_LibraryHall2_Entrance', 'GF_LibraryHall2_Centre', 10),
        ('GF_Library_Entrance', 'Staircase_GF', 15),
        ('Staircase_GF', 'Staircase_FF', 20),
        ('Staircase_FF', 'FF_ComputerCentre_Entrance', 12),
        ('Staircase_FF', 'FF_OjahCabin_Entrance', 15),
        ('Staircase_FF', 'FF_Balcony', 18),
    ]
    for src, dst, weight in edges:
        G.add_edge(src, dst, weight=weight)
        
    return G

def get_shortest_route(graph, start_node, end_node):
    try:
        path = nx.dijkstra_path(graph, start_node, end_node)
        distance = nx.dijkstra_path_length(graph, start_node, end_node)
        return path, distance
    except nx.NetworkXNoPath:
        return None, float('inf')

# ==========================================
# STEP 4: REAL-TIME NAVIGATION PIPELINE DEMO
# ==========================================
def run_navigation_demo(model, label_encoder, feature_cols, graph, dataset):
    print("\n[4/4] Executing Real-Time Indoor Navigation Simulation...")
    
    # Pick a sample fingerprint from the test set (e.g., sample reading from Ground Floor Library Entrance)
    sample_row = dataset[dataset['Clean_Location'] == 'GF_Library_Entrance'].iloc[0]
    sample_features = sample_row[feature_cols].values.reshape(1, -1)
    
    # Predict current location using trained model
    predicted_class_id = model.predict(sample_features)[0]
    current_location = label_encoder.inverse_transform([predicted_class_id])[0]
    destination_location = 'FF_ComputerCentre_Entrance'
    
    print(f"\n---> Captured Wi-Fi Fingerprint Signal Processing Complete")
    print(f"---> Predicted Current Location : {current_location}")
    print(f"---> Selected Destination       : {destination_location}")
    
    path, total_distance = get_shortest_route(graph, current_location, destination_location)
    
    if path:
        print(f"\n=== GENERATED INDOOR NAVIGATION ROUTE ===")
        print(f"Total Walking Distance: {total_distance} meters")
        print(" -> ".join(path))
        print("\nStep-by-Step Guidance:")
        for i in range(len(path) - 1):
            print(f" Step {i+1}: Walk from {path[i]} to {path[i+1]}")
    else:
        print("No valid route found between locations.")

# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
if __name__ == "__main__":
    file_path = 'WiFi_Indoor_Navigation_All_Sessions.xlsx'
    
    dataset = load_and_clean_data(file_path)
    model, label_encoder, feature_cols = evaluate_models(dataset)
    building_graph = build_building_graph()
    run_navigation_demo(model, label_encoder, feature_cols, building_graph, dataset)