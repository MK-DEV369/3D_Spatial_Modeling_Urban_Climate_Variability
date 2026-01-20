import pandas as pd
import networkx as nx
import pickle
import numpy as np
from scipy.spatial import KDTree

def get_neighboring_roads(G, locations):
    """
    Finds the names of roads connected to the nearest graph nodes for a list of coordinates.
    
    Args:
        G (nx.MultiGraph or nx.Graph): The networkx graph (typically from OSMnx).
        locations (list): A list of dicts with 'name', 'lat', and 'lng'.
        
    Returns:
        dict: A mapping of location names to a list of unique connected road names.
    """
    # 1. Convert to undirected to ensure we see all adjacent edges
    G_undirected = G.to_undirected()

    # 2. Build Spatial Index (KDTree)
    # Extract node IDs and coordinates (x=lon, y=lat)
    nodes_data = list(G_undirected.nodes(data=True))
    node_coords = []
    node_ids = []

    for nid, attrs in nodes_data:
        if 'x' in attrs and 'y' in attrs:
            node_coords.append([attrs['x'], attrs['y']])
            node_ids.append(nid)

    if not node_coords:
        raise ValueError("Graph nodes do not contain 'x' and 'y' coordinate attributes.")

    tree = KDTree(node_coords)
    results = {}

    # 3. Query Neighbors
    for loc in locations:
        # Find nearest node
        dist, idx = tree.query([loc['lng'], loc['lat']])
        nearest_node_id = node_ids[idx]
        
        neighbor_names = set()
        adj_dict = G_undirected[nearest_node_id]
        
        for neighbor_id, edges_dict in adj_dict.items():
            # Handle MultiGraph structure: {neighbor_id: {key: {attr_dict}}}
            for key, edge_data in edges_dict.items():
                name = edge_data.get('name', None)
                
                if name:
                    # OSM names can be strings or lists of strings
                    if isinstance(name, list):
                        for n in name:
                            neighbor_names.add(n)
                    else:
                        neighbor_names.add(name)
        
        results[loc['name']] = {
            "nearest_node": nearest_node_id,
            "roads": sorted(list(neighbor_names))
        }

    return results

if __name__ == "__main__":
    # 1. Load the graph
    file_path = "bengaluru_graph.pkl"
    print(f"Loading graph from {file_path}...")
    
    try:
        with open(file_path, "rb") as f:
            bengaluru_graph = pickle.load(f)
        
        # 2. Define Target List
        road_list = [
            {"name": "Sony World Junction", "lat": 12.934328, "lng": 77.623410},
            {"name": "Trinity Circle", "lat": 12.971145, "lng": 77.619743},
        ]

        # 3. Call the function
        print("Finding neighboring roads...")
        road_results = get_neighboring_roads(bengaluru_graph, road_list)

        # 4. Print Results
        print("\n--- Adjacency Results ---\n")
        for loc_name, data in road_results.items():
            print(f"📍 Location: {loc_name}")
            print(f"   Node ID: {data['nearest_node']}")
            
            if data['roads']:
                print(f"   Connected Roads: {', '.join(data['roads'])}")
            else:
                print("   Connected Roads: None found")
            print("-" * 40)

    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Please ensure the file exists.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")