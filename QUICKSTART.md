# Quick Start Guide - Urban Climate Platform

## 🚀 Start Everything (3 Simple Steps)

### Step 1: Start PostgreSQL
```powershell
# Windows Service
Start-Service postgresql-x64-18

# Check if running
Get-Service postgresql*
```

### Step 2: Start Backend
```powershell
cd backend
.\venv\Scripts\Activate.ps1
python manage.py runserver
```

**Backend ready at:** http://localhost:8000

### Step 3: Start Frontend
```powershell
# Open new terminal
cd frontend
npm run dev
```

**Frontend ready at:** http://localhost:5173

---

## 📍 Access URLs

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend** | http://localhost:5173 | Main application |
| **Dashboard** | http://localhost:5173/dashboard | Urban climate dashboard |
| **Scenario Builder** | http://localhost:5173/scenario | Create scenarios (Map + 3D) |
| **Backend API** | http://localhost:8000/api/ | REST API endpoints |
| **Admin Panel** | http://localhost:8000/admin/ | Django admin |

---

## 🏗️ First Time Setup

### 1. Install Dependencies

**Backend:**
```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Frontend:**
```powershell
cd frontend
npm install
```

### 2. Setup Database
```powershell
cd backend
.\venv\Scripts\Activate.ps1

# Run migrations
python manage.py migrate

# Create admin user (optional)
python manage.py createsuperuser

# Create sample city
python manage.py shell
```

In Python shell:
```python
from core.models import City
City.objects.create(
    name="Bengaluru",
    country="India",
    latitude=12.9716,
    longitude=77.5946,
    population=12000000
)
exit()
```

### 3. Configure CORS (Important!)
```powershell
# Install django-cors-headers
pip install django-cors-headers
```

Add to `backend/urban_climate/settings.py`:
```python
INSTALLED_APPS = [
    ...
    'corsheaders',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    ...
]

CORS_ALLOW_ALL_ORIGINS = True  # For development only
```

---

## ✅ Verify Everything Works

### Test Backend
```powershell
# Should return city data
Invoke-WebRequest http://localhost:8000/api/cities/ | ConvertFrom-Json
```

### Test Frontend
1. Open http://localhost:5173
2. Go to Dashboard or Scenario Builder
3. Select a city from dropdown
4. Map should load automatically

---

## 🎯 Key Features

### Dashboard (http://localhost:5173/dashboard)
- **Full-screen OpenStreetMap** with city visualization
- **5 Feature Tabs:**
  - 🌤️ Weather - Prediction & simulation
  - 🚗 Traffic - Congestion analysis
  - 🏙️ Urban Growth - Mathematical projections
  - 💧 Water Quality - Real-time monitoring
  - 🏗️ Buildings - Selection & removal tools

### Scenario Builder (http://localhost:5173/scenario)
- **🗺️ Map View** - 2D OpenStreetMap
- **🏙️ 3D View** - Three.js building visualization
- **Climate Overlay** - Temperature/Humidity/Precipitation
- **Scenario Management** - Create, save, delete scenarios
- **Toggle Views** - Switch between map and 3D instantly

---

## 🛠️ Common Commands

### Backend
```powershell
# Activate environment
cd backend
.\venv\Scripts\Activate.ps1

# Run server
python manage.py runserver

# Make migrations
python manage.py makemigrations
python manage.py migrate

# Django shell
python manage.py shell

# Create superuser
python manage.py createsuperuser
```

### Frontend
```powershell
cd frontend

# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

---

## 🐛 Troubleshooting

### "Map not loading"
1. Check backend is running: http://localhost:8000/api/cities/
2. Select a city from dropdown
3. Check browser console (F12) for errors
4. See [MAP_TROUBLESHOOTING.md](MAP_TROUBLESHOOTING.md)

### "No cities in dropdown"
```powershell
# Add cities via Django shell
cd backend
python manage.py shell
```
```python
from core.models import City
City.objects.create(name="Your City", country="Country", latitude=XX.XX, longitude=YY.YY)
```

### "CORS error"
- Install django-cors-headers
- Add to INSTALLED_APPS and MIDDLEWARE
- Set CORS_ALLOW_ALL_ORIGINS = True (dev only)

### "3D view not loading"
- Check city has building data: http://localhost:8000/api/cities/1/buildings/
- Check browser console for WebGL errors
- Verify city is selected

### "Port already in use"
```powershell
# Kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or use different port
python manage.py runserver 8080
```

---

## 📦 Tech Stack

**Frontend:**
- React 18 + TypeScript
- Vite (build tool)
- TanStack Query (data fetching)
- Three.js + React Three Fiber (3D)
- Tailwind CSS (styling)

**Backend:**
- Django 4.2.7
- PostgreSQL 18 + PostGIS
- Django REST Framework
- Celery (task queue)

**Map Libraries:**
- OpenStreetMap (iframe embed)
- React Three Fiber (3D buildings)
- Three.js (WebGL rendering)

---

## 📚 Documentation

- [BACKEND_SETUP.md](BACKEND_SETUP.md) - Detailed backend guide
- [MAP_TROUBLESHOOTING.md](MAP_TROUBLESHOOTING.md) - Map issues & solutions
- [FEATURES.md](FEATURES.md) - Feature documentation
- [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) - Recent changes

---

## 💡 Quick Tips

1. **Always start backend before frontend**
2. **Check PostgreSQL is running first**
3. **Configure CORS for local development**
4. **Use browser DevTools to debug map issues**
5. **Select a city to see map/3D view**
6. **Toggle between Map and 3D in ScenarioBuilder**

---

## 🎓 Learning Resources

- Django Docs: https://docs.djangoproject.com/
- React Docs: https://react.dev/
- Three.js Docs: https://threejs.org/docs/
- OpenStreetMap: https://www.openstreetmap.org/

---

**Need Help?** Check the troubleshooting guides or inspect browser console (F12) for detailed error messages.

**Status:** ✅ All systems operational  
**Last Updated:** January 17, 2026
