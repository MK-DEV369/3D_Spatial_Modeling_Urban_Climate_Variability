# Urban Climate Modeling Platform 🌍

> Visualizing the Climate Impact of Unplanned Urban Development

A comprehensive AI-powered platform that showcases how unplanned cities like Bengaluru disrupt climate patterns, increase pollution, and impact global warming — compared with planned cities like Dubai and Netherlands.

## 🎯 Project Vision

This project demonstrates that **unplanned urban development has measurable climate and economic consequences**. By visualizing Bengaluru's building density, traffic patterns, and pollution levels alongside planned cities, we provide evidence-based insights for:

- Urban planning policy decisions
- Climate change research
- Real estate development optimization
- Environmental advocacy campaigns

## ✨ Key Features

### 🏙️ **3D City Visualization**
- Interactive 3D maps powered by Three.js with WebGL acceleration
- OpenStreetMap building data with height extrusion
- Satellite imagery overlay support (MapLibre GL JS ready)
- Real-time climate data overlays (temperature, humidity, precipitation)

### 🤖 **AI-Powered Climate Modeling**
- **GraphCast**: 1-15 day weather forecasting (Google DeepMind)
- **ClimaX**: Long-term climate projection (Microsoft Research)
- **LSTM Models**: Traffic and pollution time-series prediction
- Multi-model scenario simulation

### 🌳 **Scenario Builder**
- Remove buildings and visualize climate impact
- Add vegetation/green spaces
- Real-time scenario comparison (before/after)
- Economic impact calculations

### 📊 **Multi-City Comparison**
- Compare unplanned cities: Bengaluru, Delhi, Mumbai, Chennai
- Contrast with planned cities: Dubai, Amsterdam
- Side-by-side metrics and visualizations
- Heat island effect analysis

### 🚗 **Traffic & Pollution Analysis**
- Traffic congestion modeling
- AQI (Air Quality Index) forecasting
- Pollution hotspot identification
- Economic cost of inefficient planning

## 🛠️ Technology Stack

### Frontend
- **Framework**: React 18 + TypeScript + Vite
- **3D Rendering**: Three.js, @react-three/fiber, @react-three/drei
- **Mapping**: MapLibre GL JS (for satellite overlay)
- **State Management**: React Query (TanStack Query)
- **Styling**: Tailwind CSS
- **UI Components**: Custom ReactBits animations

### Backend
- **Framework**: Django 4.2 + Django REST Framework
- **Database**: PostgreSQL 14 + PostGIS (spatial queries)
- **Task Queue**: Celery + Redis
- **API Integrations**: OpenWeatherMap, OpenAQ
- **Geospatial**: GeoDjango, Overpass API (OSM)

### ML/AI
- **GraphCast**: JAX + Haiku (weather forecasting)
- **ClimaX**: PyTorch (climate projection)
- **Traffic Model**: LSTM with PyTorch
- **Pollution Model**: Time-series forecasting

### Infrastructure
- **Containerization**: Docker + Docker Compose
- **GPU Acceleration**: NVIDIA RTX 3070 (CUDA, JAX, PyTorch)
- **Version Control**: Git

## 📁 Project Structure

```
Final_Project/
├── backend/                    # Django backend
│   ├── core/                   # Core models and business logic
│   │   ├── models.py           # City, Building, ClimateData, etc.
│   │   ├── services/           # Business logic services
│   │   │   ├── osm_service.py
│   │   │   ├── data_integration_service.py
│   │   │   ├── climate_service.py
│   │   │   ├── traffic_service.py
│   │   │   └── scenario_service.py
│   │   └── management/commands/
│   │       ├── seed_cities.py
│   │       └── sync_data.py
│   ├── api/                    # REST API endpoints
│   ├── ml_pipeline/            # ML model integrations
│   │   ├── graphcast/
│   │   ├── climax/
│   │   ├── traffic/
│   │   └── pollution/
│   └── urban_climate/          # Django settings
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Homepage/       # Landing page ✨
│   │   │   ├── Dashboard/      # Metrics dashboard
│   │   │   ├── Viewer3D/       # 3D city viewer
│   │   │   └── ScenarioBuilder/ # Scenario creation
│   │   ├── services/           # API clients
│   │   └── types/              # TypeScript types
│   └── public/
├── scripts/                    # Setup and utility scripts
└── docker-compose.yml          # Docker orchestration
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL 14+ with PostGIS
- Redis (for Celery)
- NVIDIA GPU (optional, for ML models)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env: Add database credentials, API keys (optional)

# Setup database
createdb urban_climate
psql -d urban_climate -c "CREATE EXTENSION postgis;"

# Run migrations
python manage.py migrate

# Seed cities with OSM data (takes 5-15 min)
python manage.py seed_cities --cities bengaluru delhi

# Sync weather/pollution data
python manage.py sync_data

# Create admin user
python manage.py createsuperuser

# Start server
python manage.py runserver
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

### Docker Setup (Alternative)

```bash
# Start all services
docker-compose up -d

# Seed data
docker-compose exec backend python manage.py seed_cities
docker-compose exec backend python manage.py sync_data
```

Visit:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000/api/
- **Admin**: http://localhost:8000/admin/

## 📖 Usage Guide

### 1. **Homepage**
- Learn about the project vision
- Understand the problem of unplanned cities
- Explore features and technology stack

### 2. **Dashboard**
- View real-time metrics for selected city
- Compare climate, traffic, and pollution data
- Switch between cities

### 3. **3D Viewer**
- Explore interactive 3D city map
- Toggle climate overlays (temperature, humidity)
- View building heights and density
- Select buildings for detailed info

### 4. **Scenario Builder**
- Create "what-if" scenarios
- Remove buildings to reduce density
- Add vegetation for cooling effect
- Run ML models to predict climate impact
- Compare scenarios side-by-side

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/cities/` | GET | List all cities |
| `/api/cities/{id}/` | GET | City details with buildings |
| `/api/buildings/` | GET | List buildings (with filters) |
| `/api/climate-data/` | GET | Climate data records |
| `/api/traffic-data/` | GET | Traffic data records |
| `/api/pollution-data/` | GET | Pollution/AQI data |
| `/api/scenarios/` | POST | Create new scenario |
| `/api/scenarios/{id}/run/` | POST | Run ML models for scenario |

## 🧪 Management Commands

```bash
# Seed cities
python manage.py seed_cities --cities bengaluru delhi mumbai chennai

# Skip building download (faster)
python manage.py seed_cities --skip-buildings

# Sync external data
python manage.py sync_data

# Sync specific city
python manage.py sync_data --city Bengaluru
```

## 🎓 Use Cases

### 🏛️ Urban Planning Policy
Help city planners visualize the impact of development decisions before implementation. Demonstrate the benefits of green spaces and optimized building placement.

### 📚 Academic Research
Support climate research with real-world urban modeling and scenario analysis. Provide data-driven evidence for urban heat island studies.

### 🏗️ Real Estate Development
Evaluate climate impact of proposed developments. Optimize building placement to minimize environmental footprint.

### 🌱 Environmental Advocacy
Demonstrate the importance of sustainable urban design. Visualize the economic and climate costs of poor planning.

## 📈 Scalability

### 🌐 Global Expansion
Built on OpenStreetMap data — can be extended to **any city worldwide**. Current focus: Top Indian cities + planned city comparisons.

### ⚡ Performance Optimized
- WebGL GPU acceleration (tested on RTX 3070)
- Handles thousands of buildings in real-time
- Celery async task queue for ML model execution
- Redis caching for API responses

### 🔌 Modular Architecture
- Service-oriented design
- Easy integration of new ML models
- RESTful API for third-party applications
- Plugin system for custom data sources

## 👥 Team

- **Your Name** - Full Stack Developer, ML & 3D Graphics
- **Team Member 2** - Data Scientist, Climate Modeling
- **Team Member 3** - Urban Planner, City Design

## 🗺️ Roadmap

### Phase 1: Foundation ✅ (Completed)
- [x] Backend architecture with Django + PostGIS
- [x] Frontend with React + Three.js
- [x] OSM building data integration
- [x] Basic 3D visualization
- [x] Homepage and landing page

### Phase 2: Data Integration 🚧 (In Progress)
- [x] City seeding command (6 cities)
- [x] Weather API integration (OpenWeatherMap)
- [x] Pollution API integration (OpenAQ)
- [ ] Historical data collection
- [ ] Automated daily sync (Celery periodic tasks)

### Phase 3: ML Models (Next)
- [ ] GraphCast integration
- [ ] ClimaX integration
- [ ] Traffic LSTM model training
- [ ] Pollution forecasting model

### Phase 4: Advanced Features
- [ ] MapLibre GL JS satellite overlay
- [ ] Building selection and info panels
- [ ] Vegetation addition UI
- [ ] Economic impact calculations
- [ ] Side-by-side comparison view

### Phase 5: Production
- [ ] Performance optimization
- [ ] Mobile responsiveness
- [ ] User authentication
- [ ] Cloud deployment (AWS/Azure)
- [ ] CI/CD pipeline

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **GraphCast** - Google DeepMind
- **ClimaX** - Microsoft Research
- **OpenStreetMap** - Community contributors
- **OpenWeatherMap** - Weather data API
- **OpenAQ** - Air quality data
- **Three.js** - 3D rendering library

---

**Built with ❤️ for a sustainable urban future**
