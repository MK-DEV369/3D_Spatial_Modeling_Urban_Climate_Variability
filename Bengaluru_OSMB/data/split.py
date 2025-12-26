import geopandas as gpd
import math
import os

gdf = gpd.read_file("bengaluru-buildings.geojson")

chunk_size = 5000  # buildings per file

total = len(gdf)
num_files = math.ceil(total / chunk_size)

os.makedirs("chunks", exist_ok=True)

for i in range(num_files):
    start = i * chunk_size
    end = start + chunk_size
    chunk = gdf.iloc[start:end]
    chunk.to_file(f"chunks/bengaluru_{i}.geojson", driver="GeoJSON")
    print(f"Saved chunk {i+1}/{num_files}")
