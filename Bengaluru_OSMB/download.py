import requests
import json

print("Downloading Bengaluru building data...")

# Overpass API query
overpass_url = "https://overpass-api.de/api/interpreter"
overpass_query = """
[out:json][timeout:300];
(
  way["building"](12.8,77.4,13.1,77.8);
  relation["building"](12.8,77.4,13.1,77.8);
);
out geom;
"""

response = requests.post(overpass_url, data={'data': overpass_query})

if response.status_code == 200:
    with open('bengaluru-buildings.json', 'w', encoding='utf-8') as f:
        json.dump(response.json(), f, indent=2)
    print("Download complete! File saved as bengaluru-buildings.json")
else:
    print(f"Error: {response.status_code}")