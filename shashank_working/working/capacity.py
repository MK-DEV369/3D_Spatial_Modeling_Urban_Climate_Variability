import networkx as nx
import pickle
import numpy as np

# 1. Load the Graph
# We need 'networkx' imported so Python understands the object structure during unpickling
print("Loading graph from pickle...")
with open("bengaluru_graph.pkl", "rb") as f:
    G = pickle.load(f)

# --- CONFIGURATION: CAPACITY ASSUMPTIONS ---
# 
# Theoretical Capacity per Lane (PCU/hour)
capacity_per_lane = {
    'motorway': 2000,
    'trunk': 1800,
    'primary': 1500,
    'secondary': 1200,
    'tertiary': 1000,
    'residential': 600,
    'service': 400,
    'unclassified': 800
}

# Default lanes if data is missing in the graph attributes
default_lanes = {
    'motorway': 3,
    'trunk': 2,
    'primary': 2,
    'secondary': 1,
    'tertiary': 1,
    'residential': 1,
    'service': 1,
    'unclassified': 1
}

# --- HELPER FUNCTIONS ---

def parse_lanes(lanes_val, road_type):
    """
    Safely extracts lane count, handling lists (['2', '3']) and strings ('2').
    Pure Python logic.
    """
    if lanes_val is None:
        return default_lanes.get(road_type, 1)
    
    # Handle list case: ['2', '3'] -> take max 3
    if isinstance(lanes_val, list):
        try:
            val_list = [float(x) for x in lanes_val]
            return max(val_list)
        except:
            return default_lanes.get(road_type, 1)
            
    # Handle string case: "2"
    try:
        return float(lanes_val)
    except:
        return default_lanes.get(road_type, 1)

def get_road_capacity(graph, road_name_query):
    """
    Scans the NetworkX graph edges to find roads matching the query name.
    Uses graph.edges(data=True) to access the adjacency data directly.
    """
    found_segments = []
    
    # NETWORKX EDGE ITERATION
    # graph.edges(data=True) yields (u, v, attributes_dictionary)
    # This scans the entire adjacency structure of the graph.
    for u, v, data in graph.edges(data=True):
        
        # 1. Check Name Match
        # We access the dictionary 'data' directly
        name = data.get('name', '')
        if not name:
            continue
            
        # Handle cases where name is a list (e.g. two names for one road)
        # We convert to string to search easily
        name_str = str(name).lower()
        
        if road_name_query.lower() in name_str:
            
            # 2. Extract Attributes
            highway_type = data.get('highway', 'unclassified')
            
            # If highway is a list (rare), take the first one
            if isinstance(highway_type, list):
                highway_type = highway_type[0]
            
            lanes_raw = data.get('lanes', None)
            
            # 3. Calculate Capacity
            num_lanes = parse_lanes(lanes_raw, highway_type)
            base_cap = capacity_per_lane.get(highway_type, 800)
            
            total_capacity = num_lanes * base_cap
            
            found_segments.append({
                "type": highway_type,
                "lanes": num_lanes,
                "capacity": total_capacity,
                "raw_lanes": lanes_raw
            })
    
    return found_segments

# --- EXECUTION ---

target_road = "Hosur Road"
print(f"Analyzing capacity for: {target_road}...")

segments = get_road_capacity(G, target_road)

if not segments:
    print("Road not found in graph.")
else:
    # A road consists of many small segments (edges). 
    # We average them to get the general statistics.
    avg_cap = np.mean([s['capacity'] for s in segments])
    avg_lanes = np.mean([s['lanes'] for s in segments])
    
    # Grab the most common road type in the segments found
    types = [s['type'] for s in segments]
    primary_type = max(set(types), key=types.count)
    
    print(f"\n--- Results for {target_road} ---")
    print(f"Road Type Detected: {primary_type}")
    print(f"Average Lanes: {avg_lanes:.1f}")
    print(f"Estimated Capacity: ~{int(avg_cap)} vehicles/hour")
    
    print("\nSample Segments Data (First 3 matches):")
    for i, s in enumerate(segments[:3]):
        print(f"   Segment {i+1}: Type={s['type']}, Lanes={s['lanes']}, Cap={s['capacity']}")