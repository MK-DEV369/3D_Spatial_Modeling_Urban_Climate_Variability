import geopandas as gpd
import matplotlib.pyplot as plt

gdf = gpd.read_file("bengaluru-buildings.geojson")

print(gdf.head())

gdf.plot(figsize=(10,10))
plt.show()
