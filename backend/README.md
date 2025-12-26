# Django Backend - Urban Climate Modeling API

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
# Add optional API keys:
# OPENWEATHER_API_KEY=your_key_here (for real weather data)
```

3. Set up PostgreSQL with PostGIS:
```bash
createdb urban_climate
psql -d urban_climate -c "CREATE EXTENSION postgis;"
```

4. Run migrations:
```bash
python manage.py migrate
```

5. Seed cities with OSM building data:
```bash
# Seed all cities (Bengaluru, Delhi, Mumbai, Chennai, Dubai, Amsterdam)
python manage.py seed_cities

# Or seed specific cities
python manage.py seed_cities --cities bengaluru delhi

# Skip building download (faster, creates city records only)
python manage.py seed_cities --skip-buildings
```

6. Sync external data (weather, pollution):
```bash
# Sync all cities
python manage.py sync_data

# Sync specific city
python manage.py sync_data --city Bengaluru
```

7. Create superuser:
```bash
python manage.py createsuperuser
```

8. Start development server:
```bash
python manage.py runserver
```

9. Start Celery worker (in separate terminal):
```bash
celery -A urban_climate worker --loglevel=info
```

## Management Commands

- `seed_cities` - Download OSM building data and populate database
- `sync_data` - Fetch current weather and pollution data from external APIs
- `download_osm_data` - Download OSM data for a specific city (legacy)

## Project Structure

- `urban_climate/` - Main Django project settings
- `core/` - Core business logic and models
- `api/` - REST API endpoints
- `ml_pipeline/` - ML model integration
  - `graphcast/` - Weather forecasting (GraphCast)
  - `climax/` - Climate projection (ClimaX)
  - `traffic/` - Traffic modeling (LSTM)
  - `pollution/` - Pollution forecasting
- `core/services/` - Business logic services
  - `osm_service.py` - OpenStreetMap data integration
  - `data_integration_service.py` - Weather & pollution APIs
  - `climate_service.py` - Climate modeling orchestration
  - `traffic_service.py` - Traffic prediction
  - `pollution_service.py` - Pollution analysis
  - `scenario_service.py` - Multi-model scenario simulation

## Database Models

All models are defined in `core/models.py`:
- City, Building, ClimateData, TrafficData, PollutionData, Scenario, Prediction

## API Documentation

API endpoints are defined in `api/views.py` and routed through `api/urls.py`.

## External API Integration

### OpenWeatherMap (Optional)
- Set `OPENWEATHER_API_KEY` in `.env` for real weather data
- Free tier: 1,000 calls/day, 60 calls/minute
- Without API key, system uses mock data

### OpenAQ (No key required)
- Public air quality database
- Rate limited, no authentication needed
- Falls back to mock data if unavailable

## Plan: Comprehensive Urban Climate Modeling Platform
Transform the existing Django + React foundation into a full-featured platform showcasing Bengaluru's unplanned urban sprawl and climate impact, with ML-powered scenario modeling and 3D visualization comparing unplanned vs. planned cities.

## Steps
Create homepage with project branding — Build landing page component in components with hero section, problem statement (unplanned cities → climate disruption → economic impact), interactive workflow diagram, tech stack showcase using AnimatedButton and FadeIn from reactbits, scalability section, and team details section with navigation to main app.

Seed multi-city database with real data — Write Django management command in commands to populate Bengaluru/Delhi/Mumbai/Chennai using existing osm_service.py, add planned cities (Dubai/Netherlands), import historical climate/traffic/pollution data via OpenWeatherMap/OpenAQ APIs, and create demo scenarios.

Integrate real ML models — Replace mock implementations in ml_pipeline/graphcast/inference.py and ml_pipeline/climax/inference.py with actual GraphCast (JAX) and ClimaX (PyTorch) models or use weather APIs as interim solution, train LSTM traffic/pollution models on collected data, update core/services/ to call real inference.

Add satellite imagery and advanced 3D features — Integrate Mapbox GL JS or MapLibre GL JS in Viewer3D.tsx for satellite base layer under Three.js buildings, implement building selection with info panels, add vegetation visualization (green overlays), create before/after scenario animations, and add heat map/wind flow/pollution overlays synced with ML predictions.

Build scenario manipulation and comparison — Extend ScenarioBuilder.tsx with UI for building removal (mark buildings in scenario), vegetation addition (draw green spaces), update backend scenario_service to recalculate climate/pollution with changed inputs, create side-by-side comparison view for unplanned vs. planned cities, add economic impact calculations (energy consumption, property values), and build dedicated "Unplanned City Problems" showcase page demonstrating heat island effect and traffic-pollution correlation.

Polish and deploy — Add loading states, error handling, mobile responsiveness, user tutorial, optimize Three.js rendering (frustum culling, LOD), set up Docker deployment with docker-compose for PostgreSQL/Redis/Celery/Django/React, add analytics/monitoring (Sentry), write documentation for adding new cities, and create demo walkthrough video.

Further Considerations
ML Model Availability — GraphCast and ClimaX require significant compute (8GB+ VRAM). Consider: Option A: Full integration if model weights accessible / Option B: Use OpenWeatherMap + statistical models as interim / Option C: Cloud GPU inference (AWS Lambda, Modal)

Data Pipeline Strategy — Real-time vs. batch processing? Should climate/traffic data update daily via Celery tasks, or seed historical data once for demo? API rate limits (OpenWeatherMap free tier: 1000 calls/day)?

Planned City Selection — Which planned cities best contrast with Bengaluru? Dubai (modern grid), Netherlands (canal cities with integrated green space), Singapore (vertical gardens)? Need comparable OSM building data quality.