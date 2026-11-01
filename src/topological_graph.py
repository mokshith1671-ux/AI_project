import networkx as nx

def build_campus_graph():
    """
    Builds the 10-node campus topological graph reflecting actual physical
    corridors and stairwell connections.
    """
    G = nx.Graph()

    # Physical corridor edges: (Node_A, Node_B, distance_in_meters)
    edges = [
        # Ground Floor Corridor Connections
        ('GF_LIB2_CTR', 'GF_LIB2_ENT', 8),
        ('GF_LIB2_ENT', 'GF_LIB_ENT', 12),
        ('GF_LIB_ENT', 'GF_STUD_ENT', 15),
        ('GF_STUD_ENT', 'GF_STUD_CTR', 10),
        ('GF_LIB_ENT', 'GF_STAIR', 10),       # Stairwell connects directly via Library Entrance

        # Vertical Stairwell Link (Ground Floor to First Floor)
        ('GF_STAIR', 'FF_STAIR', 15),

        # First Floor Corridor Connections
        ('FF_STAIR', 'FF_CC_ENT', 10),
        ('FF_CC_ENT', 'FF_COMP_LAB2', 12),
        ('FF_COMP_LAB2', 'FF_OJAH_ENT', 14),
    ]

    for src, dst, weight in edges:
        G.add_edge(src, dst, weight=weight)

    return G

def get_shortest_path(graph, start_node, destination_node):
    """
    Calculates Dijkstra's shortest path and generates step-by-step guidance.
    """
    path = nx.dijkstra_path(graph, start_node, destination_node)
    dist = nx.dijkstra_path_length(graph, start_node, destination_node)

    steps = []
    for i in range(len(path) - 1):
        curr_node = path[i]
        next_node = path[i + 1]
        edge_weight = graph[curr_node][next_node]['weight']

        if "STAIR" in curr_node and "STAIR" in next_node:
            steps.append(f"Take stairs from {curr_node} to {next_node} ({edge_weight}m)")
        else:
            steps.append(f"Walk from {curr_node} towards {next_node} ({edge_weight}m)")

    return path, dist, steps