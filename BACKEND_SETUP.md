# Backend Startup Guide

## Prerequisites
- Python 3.9+ installed
- PostgreSQL 18 with PostGIS extension
- Redis (for Celery task queue)
- Virtual environment activated

## Quick Start - Run Everything

### Option 1: Using Docker (Recommended)
```bash
# From project root
docker-compose up -d
```

This starts:
- PostgreSQL database (port 5432)
- Redis (port 6379)
- Django backend (port 8000)
- Celery worker

### Option 2: Manual Setup

#### 1. Start Database
```powershell
# If PostgreSQL is installed as Windows service
Start-Service postgresql-x64-18

# Or if using WSL/Linux
sudo service postgresql start
```

#### 2. Start Redis
```powershell
# If Redis is installed on Windows
redis-server

# Or if using WSL/Linux
sudo service redis-server start
```

#### 3. Activate Virtual Environment & Install Dependencies
```powershell
cd backend

# Create virtual environment (first time only)
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

#### 4. Database Setup (First Time Only)
```powershell
# Apply migrations
python manage.py migrate

# Create superuser (optional, for admin access)
python manage.py createsuperuser

# Load initial data (cities, etc.)
python manage.py loaddata initial_data.json
# OR manually create cities via admin panel
```

#### 5. Start Django Development Server
```powershell
# From backend directory
python manage.py runserver 0.0.0.0:8000
```

The API will be available at: `http://localhost:8000/api/`

#### 6. Start Celery Worker (Optional - for async tasks)
```powershell
# Open a new terminal, activate venv, then:
cd backend
celery -A urban_climate worker -l info --pool=solo
```

#### 7. Start Celery Beat (Optional - for scheduled tasks)
```powershell
# Open another terminal, activate venv, then:
cd backend
celery -A urban_climate beat -l info
```

## API Endpoints

Once the backend is running, you can access:

- **API Root**: http://localhost:8000/api/
- **Admin Panel**: http://localhost:8000/admin/
- **Cities List**: http://localhost:8000/api/cities/
- **Climate Data**: http://localhost:8000/api/cities/{id}/climate/
- **Traffic Data**: http://localhost:8000/api/cities/{id}/traffic/
- **Pollution Data**: http://localhost:8000/api/cities/{id}/pollution/
- **Buildings GeoJSON**: http://localhost:8000/api/cities/{id}/buildings/
- **Scenarios**: http://localhost:8000/api/scenarios/

## Database Configuration

### PostgreSQL Connection Settings
Edit `backend/urban_climate/settings.py`:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.contrib.gis.db.backends.postgis',
        'NAME': 'urban_climate_db',
        'USER': 'postgres',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '5432',  # or 5433 if using different port
    }
}
```

### Create Database (if not exists)
```sql
-- Connect to PostgreSQL
psql -U postgres

-- Create database
CREATE DATABASE urban_climate_db;

-- Connect to the database
\c urban_climate_db

-- Enable PostGIS
CREATE EXTENSION postgis;
```

## Environment Variables

Create a `.env` file in `backend/` directory:

```env
DEBUG=True
SECRET_KEY=your-secret-key-here
DATABASE_NAME=urban_climate_db
DATABASE_USER=postgres
DATABASE_PASSWORD=your_password
DATABASE_HOST=localhost
DATABASE_PORT=5432

# Redis (for Celery)
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# External API Keys (if needed)
OPENWEATHER_API_KEY=your_key_here
```

## Troubleshooting

### Issue: Port 8000 already in use
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

### Issue: Database connection refused
- Check if PostgreSQL is running: `Get-Service postgresql*`
- Verify port in settings.py matches your PostgreSQL port
- Check database credentials

### Issue: PostGIS extension not found
```sql
-- Install PostGIS on Windows
-- Download from: https://postgis.net/install/

-- Enable in database
psql -U postgres -d urban_climate_db -c "CREATE EXTENSION postgis;"
```

### Issue: Migrations fail
```powershell
# Reset migrations (DANGER: deletes all data)
python manage.py migrate --fake <app_name> zero
python manage.py migrate <app_name>

# Or create new migrations
python manage.py makemigrations
python manage.py migrate
```

### Issue: No cities data
```powershell
# Create cities via Django shell
python manage.py shell

# In the shell:
from core.models import City
City.objects.create(
    name="Bengaluru",
    country="India",
    latitude=12.9716,
    longitude=77.5946,
    population=12000000
)
```

## Development Workflow

### Running All Services
```powershell
# Terminal 1: Django server
cd backend
.\venv\Scripts\Activate.ps1
python manage.py runserver

# Terminal 2: Celery worker (optional)
cd backend
.\venv\Scripts\Activate.ps1
celery -A urban_climate worker -l info --pool=solo

# Terminal 3: Frontend (in separate directory)
cd frontend
npm run dev
```

### Hot Reload
Django automatically reloads when you save Python files. No need to restart the server during development.

### Running Tests
```powershell
cd backend
python manage.py test
```

### Creating Superuser for Admin Access
```powershell
python manage.py createsuperuser
# Follow prompts to set username, email, password
```

## Production Deployment

For production, use:
- **Gunicorn** instead of `runserver`
- **Nginx** as reverse proxy
- **PostgreSQL** on separate server
- **Redis** on separate server or managed service
- Environment variables for all sensitive data
- Set `DEBUG=False` in settings

Example production command:
```bash
gunicorn urban_climate.wsgi:application --bind 0.0.0.0:8000 --workers 4
```

## Common Commands Reference

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Deactivate venv
deactivate

# Install new package
pip install <package_name>
pip freeze > requirements.txt

# Make migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Run server
python manage.py runserver

# Django shell
python manage.py shell

# Collect static files (for production)
python manage.py collectstatic

# Download OSM data (custom command)
python manage.py download_osm_data --city Bengaluru
```

## Frontend Connection

The frontend connects to backend via API calls to `http://localhost:8000/api/`.

If backend is running on different host/port, update:
- `frontend/src/services/api.ts` - API_BASE_URL constant

## Health Check

Test if backend is running:
```powershell
# Using PowerShell
Invoke-WebRequest http://localhost:8000/api/ | Select-Object StatusCode

# Using curl (if installed)
curl http://localhost:8000/api/

# Expected output: 200 OK with API endpoints list
```

---

**Last Updated**: January 17, 2026  
**Backend Framework**: Django 4.2.7  
**Database**: PostgreSQL 18 + PostGIS  
**Task Queue**: Celery + Redis
