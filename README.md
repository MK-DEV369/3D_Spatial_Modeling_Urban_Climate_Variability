# 3D Spatial Modeling of Urban Climate Variability

A comprehensive system for modeling and visualizing urban climate variability using Django REST API backend and React + TypeScript frontend.

## Project Structure

```
Final_Project/
├── backend/              # Django backend with PostgreSQL + PostGIS
├── frontend/             # React + TypeScript frontend with Vite
└── Bengaluru_OSMB/       # Existing OSM data processing scripts
```

## OSM Data Download (Context)

This project uses OpenStreetMap (OSM) data for southern India to provide detailed building and infrastructure information for urban climate modeling. The OSM data can be downloaded from the official Geofabrik portal:

- [Geofabrik Asia/India OSM extracts](https://download.geofabrik.de/asia/india/)

For this project, the file `southern-zone.osm.pbf` was downloaded to represent the southern region of India. This file is used as a data source for further processing and integration into the backend and ML pipelines.

Example download command:

```bash
wget https://download.geofabrik.de/asia/india/south-india-latest.osm.pbf -O southern-zone.osm.pbf
```

Place the downloaded file in the project root or the appropriate data directory as required by your workflow.

## Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL 14+ with PostGIS extension
- Redis (for Celery task queue)

## Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up PostgreSQL database with PostGIS:
```bash
# Create database
createdb urban_climate

# Enable PostGIS extension
psql -d urban_climate -c "CREATE EXTENSION postgis;"
```

5. Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
# Edit .env with your database credentials
```

6. Run migrations:
```bash
python manage.py migrate
```

7. Create a superuser:
```bash
python manage.py createsuperuser
```

8. Start the development server:
```bash
python manage.py runserver
```

9. In a separate terminal, start Celery worker:
```bash
celery -A urban_climate worker --loglevel=info
```

10. Start Redis (if not running):
```bash
redis-server
```

## Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Install ReactBits.dev components (optional, for enhanced UI):
```bash
# Install jsrepo CLI globally
npm install -g jsrepo

# Initialize ReactBits with TypeScript + Tailwind CSS
npx jsrepo init https://reactbits.dev/ts/tailwind/

# Add specific components as needed
npx jsrepo add https://reactbits.dev/ts/tailwind/Buttons/AnimatedButton
npx jsrepo add https://reactbits.dev/ts/tailwind/Cards/HoverCard
npx jsrepo add https://reactbits.dev/ts/tailwind/TextAnimations/SplitText
npx jsrepo add https://reactbits.dev/ts/tailwind/Animations/FadeIn
```

4. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## API Endpoints

- `GET /api/cities/` - List all cities
- `GET /api/cities/{id}/` - Get city details
- `GET /api/cities/{id}/buildings/` - Get buildings GeoJSON for a city
- `GET /api/cities/{id}/climate/` - Get historical climate data
- `GET /api/cities/{id}/traffic/` - Get historical traffic data
- `GET /api/cities/{id}/pollution/` - Get historical pollution data
- `GET /api/scenarios/` - List all scenarios
- `POST /api/scenarios/` - Create a new scenario
- `GET /api/scenarios/{id}/predictions/` - Get predictions for a scenario
- `POST /api/scenarios/{id}/run/` - Run scenario simulation

## Database Models

- **City**: City information with geographic bounds
- **Building**: OSM building data with PostGIS geometry
- **ClimateData**: Historical and predicted climate data
- **TrafficData**: Historical and predicted traffic data
- **PollutionData**: Historical and predicted pollution data
- **Scenario**: User-defined climate scenarios
- **Prediction**: ML model predictions

## Development

### Backend
- Django admin: `http://localhost:8000/admin/`
- API root: `http://localhost:8000/api/`

### Frontend
- Development server: `http://localhost:5173`
- Hot module replacement enabled

## Next Steps

1. Seed initial city data (Bengaluru, Delhi, Mumbai, Chennai)
2. Integrate OSM data processing service
3. Set up ML pipeline integration (GraphCast, ClimaX)
4. Implement traffic and pollution prediction models
5. Enhance UI with ReactBits components
6. Add 3D visualization enhancements

## License

MIT

