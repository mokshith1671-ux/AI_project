import streamlit as st
import pandas as pd
import numpy as np
import joblib
import networkx as nx
import matplotlib.pyplot as plt
from src.topological_graph import build_campus_graph, get_shortest_path

st.set_page_config(page_title="AI Indoor Navigation", layout="wide")

st.title("🏢 AI-Based Indoor Navigation System")
st.caption("Wi-Fi RSSI Fingerprinting & Multi-Floor Topological Pathfinding")

@st.cache_resource
def load_artifacts():
    floor_model = joblib.load("models/floor_model.pkl")
    rp_model = joblib.load("models/rp_model.pkl")
    graph = build_campus_graph()
    return floor_model, rp_model, graph

try:
    floor_model, rp_model, graph = load_artifacts()
    st.sidebar.success("Models & Graph Artifacts Loaded!")
except Exception as e:
    st.sidebar.error("Error loading model files in `models/`. Run `python main.py` first.")
    st.stop()

st.sidebar.header("🕹️ Location Controls")

all_rps = list(graph.nodes())

# Start Node Selection (Current Location)
start_mode = st.sidebar.radio(
    "Current Location Mode:", 
    ["Manual Selection", "Predict from Wi-Fi Signal"]
)

if start_mode == "Manual Selection":
    start_node = st.sidebar.selectbox(
        "Current Location (Start Node)", 
        options=all_rps, 
        index=all_rps.index("GF_STUD_CTR")
    )
else:
    start_node = None

# Destination Selection
destination = st.sidebar.selectbox(
    "Destination Room / Node", 
    options=all_rps, 
    index=all_rps.index("GF_STAIR")
)

st.subheader("1. Location & Route Generation")

# Execute Route Calculation
if start_mode == "Predict from Wi-Fi Signal":
    # Simulated 19 AP RSSI input vector
    input_rssi = np.full((1, 19), -100.0)
    input_rssi[0, :3] = [-60, -65, -72]  # Mock RSSI near GF_STUD_CTR
    
    pred_floor = floor_model.predict(input_rssi)[0]
    start_node = rp_model.predict(input_rssi)[0]
    st.success(f"**Predicted Current Location:** {start_node} ({pred_floor})")
else:
    st.info(f"**Current Location (Selected):** {start_node}")

if start_node == destination:
    st.warning("Current location and destination are the same!")
else:
    path, dist, steps = get_shortest_path(graph, start_node, destination)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🗺️ Step-by-Step Directions")
        st.metric("Total Walking Distance", f"{dist} meters")
        for idx, step in enumerate(steps, 1):
            st.write(f"**Step {idx}:** {step}")
            
    with col2:
        st.markdown("### 📌 Topological Route Graph")
        fig, ax = plt.subplots(figsize=(7, 5))
        pos = nx.spring_layout(graph, seed=42)
        
        path_edges = list(zip(path[:-1], path[1:]))
        
        # Base Nodes & Edges
        nx.draw_networkx_nodes(graph, pos, node_color='lightgray', node_size=700, ax=ax)
        nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), edge_color='gray', ax=ax)
        
        # Path & Highlights
        nx.draw_networkx_nodes(graph, pos, nodelist=path, node_color='limegreen', node_size=800, ax=ax)
        nx.draw_networkx_nodes(graph, pos, nodelist=[start_node], node_color='dodgerblue', node_size=900, ax=ax)
        nx.draw_networkx_nodes(graph, pos, nodelist=[destination], node_color='crimson', node_size=900, ax=ax)
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='green', width=3, ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=8, font_weight='bold', ax=ax)
        
        plt.axis('off')
        st.pyplot(fig)