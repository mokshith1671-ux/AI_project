import os
import pandas as pd
import numpy as np
import networkx as nx
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ==========================================
# STEP 1: LOAD BENCHMARK DATASET
# ==========================================
def find_csv(filename, search_dir="data"):
    matches = glob.glob(f"{search_dir}/**/{filename}", recursive=True)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {filename} inside {search_dir}")

def load_data():
    print("[1/4] Loading UJIIndoorLoc Training and Validation Datasets...")
    train_path = find_csv("trainingData.csv")
    val_path = find_csv("validationData.csv")
    
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    print(f"Training samples  : {df_train.shape[0]}")
    print(f"Validation samples: {df_val.shape[0]}")
    
    # Extract WAP features (WAP001 to WAP520)
    wap_cols = [c for c in df_train.columns if c.startswith("WAP")]
    
    # Preprocess missing signals: UJIIndoorLoc uses +100 for unobserved APs -> convert to -105 dBm
    X_train = df_train[wap_cols].replace(100, -105).values
    X_val = df_val[wap_cols].replace(100, -105).values
    
    # Targets: Floor prediction within Building 0
    # Filter to Building 0 for focused floor-level localization navigation
    b0_train_mask = df_train['BUILDINGID'] == 0
    b0_val_mask = df_val['BUILDINGID'] == 0
    
    X_train_b0 = X_train[b0_train_mask]
    y_train_b0 = df_train.loc[b0_train_mask, 'FLOOR'].values
    
    X_val_b0 = X_val[b0_val_mask]
    y_val_b0 = df_val.loc[b0_val_mask, 'FLOOR'].values
    
    print(f"\nBuilding 0 Filtered Samples -> Train: {len(X_train_b0)}, Validation: {len(X_val_b0)}")
    return X_train_b0, y_train_b0, X_val_b0, y_val_b0

# ==========================================
# STEP 2: ML & DL MODEL BENCHMARK
# ==========================================
def evaluate_benchmark_models(X_train, y_train, X_val, y_val):
    print("\n[2/4] Benchmarking ML & DL Architectures on UJIIndoorLoc...")
    
    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost Classifier": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred) * 100
        f1 = f1_score(y_val, y_pred, average='macro')
        results[name] = (model, acc, f1)
        print(f" -> {name:22s} | Validation Accuracy: {acc:6.2f}% | Macro F1: {f1:.4f}")
        
    best_name = max(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]
    print(f"\nTop Performing Model: {best_name} ({results[best_name][1]:.2f}% Accuracy)")
    return best_model, best_name

# ==========================================
# STEP 3: INDOOR GRAPH & PATHFINDING
# ==========================================
def build_multi_floor_graph():
    print("\n[3/4] Building Multi-Floor Graph for Building 0...")
    G = nx.Graph()
    
    # Floors 0 through 3 with stairs/elevator transitions
    edges = [
        # Floor 0 Nodes
        ('B0_F0_Entrance', 'B0_F0_Hallway', 15),
        ('B0_F0_Hallway', 'B0_F0_Stairs', 10),
        # Floor 1 Nodes
        ('B0_F0_Stairs', 'B0_F1_Stairs', 12),  # Vertical stair transition
        ('B0_F1_Stairs', 'B0_F1_Hallway', 10),
        ('B0_F1_Hallway', 'B0_F1_Lab_101', 8),
        # Floor 2 Nodes
        ('B0_F1_Stairs', 'B0_F2_Stairs', 12),  # Vertical stair transition
        ('B0_F2_Stairs', 'B0_F2_ConferenceRoom', 15),
    ]
    
    for src, dst, w in edges:
        G.add_edge(src, dst, weight=w)
    return G

# ==========================================
# STEP 4: NAVIGATION SIMULATION
# ==========================================
def run_navigation_sim(model, X_val, y_val, graph):
    print("\n[4/4] Simulating Multi-Floor Indoor Navigation...")
    
    # Pick a validation sample
    sample_idx = 5
    sample_signal = X_val[sample_idx].reshape(1, -1)
    actual_floor = y_val[sample_idx]
    
    predicted_floor = model.predict(sample_signal)[0]
    
    current_node = f"B0_F{predicted_floor}_Stairs"
    destination_node = "B0_F2_ConferenceRoom"
    
    print(f"---> Real Validation RSSI Signal Processed")
    print(f"---> Actual Floor    : Floor {actual_floor}")
    print(f"---> Predicted Floor : Floor {predicted_floor}")
    print(f"---> Current Location Node : {current_node}")
    print(f"---> Destination Node      : {destination_node}")
    
    path = nx.dijkstra_path(graph, current_node, destination_node)
    dist = nx.dijkstra_path_length(graph, current_node, destination_node)
    
    print(f"\n=== GENERATED MULTI-FLOOR ROUTE ===")
    print(f"Total Distance: {dist} meters")
    print(" -> ".join(path))

if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_data()
    best_model, best_name = evaluate_benchmark_models(X_train, y_train, X_val, y_val)
    graph = build_multi_floor_graph()
    run_navigation_sim(best_model, X_val, y_val, graph)