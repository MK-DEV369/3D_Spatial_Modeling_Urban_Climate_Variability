#!/bin/bash
# Backend setup script

echo "Setting up Django backend..."

cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
venv/Scripts/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "Please edit .env file with your database credentials"
fi

echo "Backend setup complete!"
echo "Next steps:"
echo "1. Edit backend/.env with your database credentials"
echo "2. Create PostgreSQL database: createdb urban_climate"
echo "3. Enable PostGIS: psql -d urban_climate -c 'CREATE EXTENSION postgis;'"
echo "4. Run migrations: python manage.py migrate"
echo "5. Create superuser: python manage.py createsuperuser"
echo "6. Start server: python manage.py runserver"

