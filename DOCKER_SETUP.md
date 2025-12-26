# Docker Setup Guide for Urban Climate Project

## Quick Start

### Prerequisites
- Docker Desktop installed and running
- Docker Compose (included with Docker Desktop)
- No local PostgreSQL needed—everything runs in containers

### Start the Project

1. Navigate to the project root:
```bash
cd e:\5th SEM Data\MainEL\Final_Project
```

2. Start all services:
```bash
docker-compose up -d
```

3. Wait for services to be healthy (20-30 seconds):
```bash
docker-compose ps
```

All services should show `healthy` or `running`.

### Initialize the Database

The database migrations run automatically on backend startup, but if you need to manually run them:

```bash
docker-compose exec backend python manage.py migrate
```

### Create a Superuser

```bash
docker-compose exec backend python manage.py createsuperuser
```

Follow the prompts to create an admin account.

### Access the Application

- **Django Admin**: http://localhost:8000/admin/
- **API Root**: http://localhost:8000/api/
- **Frontend**: http://localhost:5173 (if you run it separately)

## Services

### PostgreSQL + PostGIS
- **Container**: `urban_climate_db`
- **Port**: 5432
- **Database**: urban_climate
- **User**: postgres
- **Password**: postgres
- **Volume**: `postgres_data` (persistent)

### Redis
- **Container**: `urban_climate_redis`
- **Port**: 6379
- **Volume**: `redis_data` (persistent)

### Django Backend
- **Container**: `urban_climate_backend`
- **Port**: 8000
- **Auto-runs migrations on startup**

### Celery Worker
- **Container**: `urban_climate_celery`
- **Processes async tasks** (weather, climate, traffic, pollution predictions)
- **Depends on**: db, redis, backend

## Useful Commands

### View logs
```bash
docker-compose logs -f backend          # Backend logs
docker-compose logs -f celery           # Celery worker logs
docker-compose logs -f db               # Database logs
docker-compose logs -f redis            # Redis logs
```

### Access database via psql
```bash
docker-compose exec db psql -U postgres -d urban_climate
```

### Stop services
```bash
docker-compose stop
```

### Remove all containers and volumes (cleanup)
```bash
docker-compose down -v
```

### Rebuild containers
```bash
docker-compose down
docker-compose up -d --build
```

## Troubleshooting

### Port already in use
If port 5432 (PostgreSQL) or 8000 (Django) is in use, edit `docker-compose.yml`:
```yaml
ports:
  - "5433:5432"  # Use 5433 instead of 5432
  - "8001:8000"  # Use 8001 instead of 8000
```

Then update your `.env` file with the new database host/port.

### Database connection refused
Wait for PostgreSQL to be healthy:
```bash
docker-compose ps
```

If `db` shows `unhealthy`, check logs:
```bash
docker-compose logs db
```

### Celery not processing tasks
Ensure Redis is running:
```bash
docker-compose logs redis
```

## Frontend Setup

The docker-compose.yml focuses on the backend. For frontend development:

1. In a separate terminal, navigate to `frontend/`:
```bash
cd frontend
npm install
npm run dev
```

Frontend will run on http://localhost:5173 and connect to `http://localhost:8000/api/`.

## Environment Variables

The compose file uses these defaults—to customize, edit `.env.docker` and run:
```bash
docker-compose --env-file .env.docker up -d
```

Or modify `docker-compose.yml` directly under the `environment:` sections.

## Notes

- **Development mode**: DEBUG=True is enabled by default
- **Production**: Change SECRET_KEY and set DEBUG=False before deploying
- **GDAL/GEOS**: Pre-installed in the Docker image—no local setup needed
- **Database persistence**: Data persists in named volumes even if containers are removed
