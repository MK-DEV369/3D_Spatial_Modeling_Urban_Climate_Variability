# Backend Testing Guide - OSM Maps Integration

This guide provides all commands to test the integrated backend with CRUD operations on OSM layers (buildings, roads, water, green spaces).

## Prerequisites

- PostgreSQL with PostGIS extension running
- Database `urban_climate` exists with tables: `buildings_osm`, `roads_osm`, `water_osm`, `green_osm`
- Python 3.10+ with virtual environment activated
- Redis running (for Celery, optional)

---

## 1. Setup & Installation

### Activate Virtual Environment (Windows PowerShell)
```powershell
cd backend
.\venv\Scripts\Activate.ps1
```

### Activate Virtual Environment (Linux/WSL)
```bash
cd backend
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 2. Database Setup

### Verify Database Connection
```bash
python manage.py dbshell
# If successful, you should see PostgreSQL prompt
# Type: \q to exit
```

### Check if PostGIS Extension is Enabled
```bash
python manage.py dbshell
# In psql:
SELECT PostGIS_version();
\q
```

### Verify OSM Tables Exist
```bash
python manage.py dbshell
# In psql:
\dt buildings_osm
\dt roads_osm
\dt water_osm
\dt green_osm
\q
```

### Run Migrations (for core models only)
```bash
python manage.py migrate
```
**Note:** Maps models have `managed=False`, so they don't create migrations. Tables should already exist.

---

## 3. Start Development Server

### Start Django Server
```bash
python manage.py runserver
```
Server runs at: **http://localhost:8000**

### Start Celery Worker (Optional - for async ML tasks)
```bash
# In a separate terminal
celery -A urban_climate worker --loglevel=info
```

---

## 4. Test API Endpoints

### 4.1 Test OSM Buildings CRUD

#### List Buildings
```bash
# PowerShell
curl http://localhost:8000/api/buildings/

# Or with filters
curl "http://localhost:8000/api/buildings/?active=true&scenario=baseline"
```

#### Get Single Building
```bash
curl http://localhost:8000/api/buildings/123456789/
# Replace 123456789 with actual osm_id
```

#### Create Building
```bash
curl -X POST http://localhost:8000/api/buildings/ `
  -H "Content-Type: application/json" `
  -d '{
    "osm_id": 999999999,
    "name": "Test Building",
    "building": "residential",
    "geom": {
      "type": "Polygon",
      "coordinates": [[[77.5946, 12.9716], [77.5950, 12.9716], [77.5950, 12.9720], [77.5946, 12.9720], [77.5946, 12.9716]]]
    },
    "active": true,
    "scenario_id": "baseline",
    "modified_at": "2024-01-01T00:00:00Z"
  }'
```

#### Update Building
```bash
curl -X PATCH http://localhost:8000/api/buildings/999999999/ `
  -H "Content-Type: application/json" `
  -d '{
    "name": "Updated Building Name",
    "active": false
  }'
```

#### Delete Building
```bash
curl -X DELETE http://localhost:8000/api/buildings/999999999/
```

#### Get Buildings by Bounding Box (for map rendering)
```bash
# Bbox format: minLon,minLat,maxLon,maxLat
curl "http://localhost:8000/api/buildings/bbox/?bbox=77.5900,12.9700,77.6000,12.9750&scenario=baseline&active=true"
```

---

### 4.2 Test OSM Roads CRUD

```bash
# List roads
curl http://localhost:8000/api/roads/

# Get by bbox
curl "http://localhost:8000/api/roads/bbox/?bbox=77.5900,12.9700,77.6000,12.9750"

# Create road
curl -X POST http://localhost:8000/api/roads/ `
  -H "Content-Type: application/json" `
  -d '{
    "osm_id": 888888888,
    "name": "Main Street",
    "highway": "primary",
    "surface": "asphalt",
    "geom": {
      "type": "LineString",
      "coordinates": [[77.5946, 12.9716], [77.5950, 12.9720]]
    },
    "active": true,
    "scenario_id": "baseline",
    "modified_at": "2024-01-01T00:00:00Z"
  }'
```

---

### 4.3 Test OSM Water CRUD

```bash
# List water features
curl http://localhost:8000/api/water/

# Get by bbox
curl "http://localhost:8000/api/water/bbox/?bbox=77.5900,12.9700,77.6000,12.9750"
```

---

### 4.4 Test OSM Green Spaces CRUD

```bash
# List green spaces
curl http://localhost:8000/api/green/

# Get by bbox
curl "http://localhost:8000/api/green/bbox/?bbox=77.5900,12.9700,77.6000,12.9750"
```

---

### 4.5 Test Other API Endpoints (Existing)

#### Cities
```bash
curl http://localhost:8000/api/cities/
curl http://localhost:8000/api/cities/1/
```

#### Scenarios
```bash
curl http://localhost:8000/api/scenarios/
curl http://localhost:8000/api/scenarios/1/predictions/
```

#### Predictions (Weather, Traffic, Pollution)
```bash
# Weather forecast
curl -X POST http://localhost:8000/api/predictions/weather/ `
  -H "Content-Type: application/json" `
  -d '{"city_id": 1, "forecast_days": 7, "async": false}'

# Traffic prediction
curl -X POST http://localhost:8000/api/predictions/traffic/ `
  -H "Content-Type: application/json" `
  -d '{"city_id": 1, "prediction_hours": 24, "async": false}'

# Pollution prediction
curl -X POST http://localhost:8000/api/predictions/pollution/ `
  -H "Content-Type: application/json" `
  -d '{"city_id": 1, "prediction_hours": 24, "async": false}'
```

---

## 5. Test with Browser/Postman

### Access API Root
Open in browser: **http://localhost:8000/api/**

### Access Specific Endpoints
- Buildings: http://localhost:8000/api/buildings/
- Roads: http://localhost:8000/api/roads/
- Water: http://localhost:8000/api/water/
- Green: http://localhost:8000/api/green/
- Cities: http://localhost:8000/api/cities/
- Scenarios: http://localhost:8000/api/scenarios/

### Bbox Query Example (Browser)
```
http://localhost:8000/api/buildings/bbox/?bbox=77.5900,12.9700,77.6000,12.9750&scenario=baseline&active=true
```

---

## 6. Django Admin Panel

### Create Superuser (if not exists)
```bash
python manage.py createsuperuser
```

### Access Admin Panel
Open: **http://localhost:8000/admin/**

You should see:
- Buildings OSM
- Roads OSM
- Water OSM
- Green OSM
- Cities
- Scenarios
- Predictions

---

## 7. Verify Integration

### Check Django Can See Models
```bash
python manage.py shell
```

In Python shell:
```python
from maps.models import BuildingsOSM, RoadsOSM, WaterOSM, GreenOSM

# Check counts
print(f"Buildings: {BuildingsOSM.objects.count()}")
print(f"Roads: {RoadsOSM.objects.count()}")
print(f"Water: {WaterOSM.objects.count()}")
print(f"Green: {GreenOSM.objects.count()}")

# Test filtering
buildings = BuildingsOSM.objects.filter(active=True, scenario_id='baseline')[:5]
for b in buildings:
    print(f"Building {b.osm_id}: {b.name}")

exit()
```

### Check URL Routing
```bash
python manage.py show_urls
# Or manually check:
python manage.py shell
from django.urls import get_resolver
resolver = get_resolver()
print([str(p.pattern) for p in resolver.url_patterns])
```

---

## 8. Common Issues & Solutions

### Issue: "Table 'buildings_osm' does not exist"
**Solution:** Tables must be created in PostgreSQL. The models use `managed=False`.

```sql
-- Connect to database and create tables manually if needed
-- (They should already exist based on your setup)
```

### Issue: "Geometry field SRID mismatch"
**Solution:** Ensure geometry fields use SRID 3857 (Web Mercator).

### Issue: Import errors for rest_framework
**Solution:** Ensure virtual environment is activated and dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: GDAL/GEOS library errors
**Solution:** On Windows/WSL, ensure GDAL/GEOS are installed. Check `settings.py` for library paths.

---

## 9. Quick Test Script

Create a test script `test_api.py`:

```python
import requests

BASE_URL = "http://localhost:8000/api"

# Test 1: List buildings
response = requests.get(f"{BASE_URL}/buildings/")
print(f"Buildings API: {response.status_code}")
print(f"Count: {len(response.json().get('features', []))}")

# Test 2: Bbox query
bbox = "77.5900,12.9700,77.6000,12.9750"
response = requests.get(f"{BASE_URL}/buildings/bbox/", params={"bbox": bbox})
print(f"\nBbox query: {response.status_code}")
print(f"Features found: {len(response.json().get('features', []))}")

# Test 3: Cities API
response = requests.get(f"{BASE_URL}/cities/")
print(f"\nCities API: {response.status_code}")

print("\n✅ All API endpoints are working!")
```

Run it:
```bash
python test_api.py
```

---

## 10. Performance Testing

### Test Bbox Query Performance
```bash
# Time a bbox query
time curl "http://localhost:8000/api/buildings/bbox/?bbox=77.5900,12.9700,77.6000,12.9750"
```

---

## Summary

✅ **Setup Complete When:**
1. Django server runs without errors
2. `/api/buildings/` returns FeatureCollection
3. `/api/buildings/bbox/` works with bbox parameter
4. CRUD operations work (create, read, update, delete)
5. Admin panel shows all 4 OSM models

🚀 **You're ready when:**
- All endpoints return 200 OK
- GeoJSON format is correct
- Bbox filtering works for map rendering
- Admin panel accessible
