# Urban Climate Platform: OSM Building & Map Operations Guide

## 1. Removing Buildings from OpenStreetMap (OSM)

### Steps:
1. **Download OSM Buildings Data**
   - Use [Overpass Turbo](https://overpass-turbo.eu/) or [Geofabrik](https://download.geofabrik.de/) to export building footprints for your city (e.g., Bengaluru).
   - Example Overpass query for buildings:
     ```
     [out:json][timeout:25];
     area[name="Bengaluru"]->.searchArea;
     (node["building"](area.searchArea);way["building"](area.searchArea);relation["building"](area.searchArea););
     out body;
     >;
     out skel qt;
     ```
   - Save as GeoJSON for easy processing.

2. **Visualize & Select Buildings**
   - Load GeoJSON in your frontend (e.g., using Leaflet, Mapbox, or Three.js).
   - Implement selection tools (rectangle, polygon, etc.) for users to choose buildings to remove.

3. **Remove Selected Buildings**
   - Filter out selected building features from the GeoJSON.
   - Re-render the map without these buildings.
   - Optionally, save the modified GeoJSON for future use.

## 2. Faster Map Loading
- Use vector tiles (e.g., Mapbox, OpenMapTiles) for scalable, fast rendering.
- Limit data to only required features (buildings, lakes, etc.).
- Use server-side preprocessing to clip and simplify geometries.
- Lazy-load map layers and use efficient rendering libraries (e.g., deck.gl, three.js).

## 3. Deleting Roads from Maps
- Download OSM data for roads using Overpass Turbo (similar to buildings).
- Remove or filter out road features ("highway" tag) from your GeoJSON before rendering.
- Example Overpass query for roads:
  ```
  [out:json][timeout:25];
  area[name="Bengaluru"]->.searchArea;
  (way["highway"](area.searchArea););
  out body;
  >;
  out skel qt;
  ```

## 4. Major Cities of India: Example Coordinates & Lakes

| City        | Latitude  | Longitude | Major Lakes (Coordinates)                |
|-------------|-----------|-----------|------------------------------------------|
| Bengaluru   | 12.9716   | 77.5946   | Ulsoor: 12.9882,77.6190; Bellandur: 12.9352,77.6784; Hebbal: 13.0368,77.5970 |
| Mumbai      | 19.0760   | 72.8777   | Powai: 19.1202,72.9056; Vihar: 19.1551,72.9028; Tulsi: 19.1802,72.8936       |
| Delhi       | 28.6139   | 77.2090   | Sanjay Lake: 28.6016,77.2952; Hauz Khas: 28.5535,77.1926                    |
| Hyderabad   | 17.3850   | 78.4867   | Hussain Sagar: 17.4239,78.4738; Osman Sagar: 17.3432,78.3212                 |
| Chennai     | 13.0827   | 80.2707   | Chembarambakkam: 13.0107,80.0267; Red Hills: 13.1700,80.1840                 |

## 5. Should You Use GraphML?
- **GraphML** is useful for network/graph analysis (e.g., road networks, connectivity).
- For building/lake operations, GeoJSON is simpler and widely supported.
- Use GraphML if you need to analyze relationships (e.g., shortest path, centrality) between map features.

## 6. How to Get City Data (e.g., Bengaluru)
- **Recommended Free Datasets:**
  - [Geofabrik](https://download.geofabrik.de/): OSM extracts for Indian cities.
  - [OpenStreetMap](https://www.openstreetmap.org/): Direct downloads and Overpass queries.
  - [Bhuvan](https://bhuvan.nrsc.gov.in/): Indian government geospatial portal (for additional layers).
  - [Data.gov.in](https://data.gov.in/): Various open datasets for Indian cities.

- **Steps:**
  1. Download OSM extract (PBF or GeoJSON) for your city.
  2. Use tools like osmconvert, osmfilter, or QGIS to extract buildings, lakes, roads, etc.
  3. Convert to GeoJSON for frontend use.

## 7. Example: Downloading Bengaluru Buildings
- Use Overpass Turbo with the query above, or download from Geofabrik (India > Karnataka > Bengaluru).
- Convert to GeoJSON if needed.
- Load into your app for selection/removal operations.

---

**References:**
- [Overpass Turbo](https://overpass-turbo.eu/)
- [Geofabrik Downloads](https://download.geofabrik.de/)
- [OpenStreetMap Wiki](https://wiki.openstreetmap.org/wiki/Main_Page)
- [Bhuvan Portal](https://bhuvan.nrsc.gov.in/)
- [Data.gov.in](https://data.gov.in/)
