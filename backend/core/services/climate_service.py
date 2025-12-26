"""
Climate prediction service that orchestrates GraphCast and ClimaX models.

This service combines short-term weather forecasts (GraphCast) with
long-term climate projections (ClimaX) to provide comprehensive
climate predictions for urban areas.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from django.utils import timezone
from core.models import City, ClimateData, Scenario, Prediction
from ml_pipeline.graphcast.inference import predict_weather, get_graphcast_instance
from ml_pipeline.climax.inference import predict_climate, get_climax_instance

logger = logging.getLogger(__name__)


def get_initial_conditions(city: City, days_back: int = 7) -> Dict:
    """
    Get initial weather conditions from historical data.
    
    Args:
        city: City model instance
        days_back: Number of days to look back for initial conditions
    
    Returns:
        Dictionary with initial conditions
    """
    cutoff_date = timezone.now() - timedelta(days=days_back)
    recent_data = ClimateData.objects.filter(
        city=city,
        timestamp__gte=cutoff_date
    ).order_by('-timestamp')[:days_back]
    
    if not recent_data.exists():
        # Use default values if no historical data
        return {
            'temperature': 25.0,
            'humidity': 60.0,
            'pressure': 1013.25,
            'wind_speed': 5.0,
        }
    
    # Use most recent data point
    latest = recent_data.first()
    return {
        'temperature': latest.temperature,
        'humidity': latest.humidity or 60.0,
        'pressure': latest.pressure or 1013.25,
        'wind_speed': latest.wind_speed or 5.0,
        'wind_direction': latest.wind_direction,
        'precipitation': latest.precipitation or 0.0,
    }


def get_historical_climate_data(city: City, years_back: int = 5) -> List[Dict]:
    """
    Get historical climate data for ClimaX input.
    
    Args:
        city: City model instance
        years_back: Number of years of historical data to retrieve
    
    Returns:
        List of climate data dictionaries
    """
    cutoff_date = timezone.now() - timedelta(days=years_back * 365)
    historical = ClimateData.objects.filter(
        city=city,
        timestamp__gte=cutoff_date
    ).order_by('timestamp')
    
    data = []
    for record in historical:
        data.append({
            'timestamp': record.timestamp.isoformat(),
            'temperature': record.temperature,
            'humidity': record.humidity,
            'precipitation': record.precipitation or 0.0,
            'wind_speed': record.wind_speed,
            'pressure': record.pressure,
        })
    
    return data


def get_urban_features(city: City) -> Dict:
    """
    Extract urban features from city data for climate modeling.
    
    Args:
        city: City model instance
    
    Returns:
        Dictionary with urban features
    """
    # Calculate building density
    building_count = city.buildings.count()
    if city.bounds:
        # Calculate area in square kilometers (rough estimate)
        # In production, use proper area calculation
        area_km2 = 100  # Placeholder
        building_density = building_count / area_km2 if area_km2 > 0 else 0
    else:
        building_density = 0
    
    # Get average building height
    from django.db.models import Avg
    buildings_with_height = city.buildings.exclude(height__isnull=True)
    if buildings_with_height.exists():
        avg_height = buildings_with_height.aggregate(
            avg=Avg('height')
        )['avg'] or 0
    else:
        avg_height = 0
    
    return {
        'building_density': building_density,
        'average_building_height': avg_height,
        'vegetation_coverage': city.metadata.get('vegetation_coverage', 20.0),  # Percentage
    }


def generate_weather_forecast(
    city: City,
    forecast_days: int = 7,
    model_path: Optional[str] = None,
) -> Dict:
    """
    Generate short-term weather forecast using GraphCast (1-15 days).
    
    Args:
        city: City model instance
        forecast_days: Number of days to forecast (1-15)
        model_path: Optional path to GraphCast model weights
    
    Returns:
        Dictionary with forecast results
    """
    if forecast_days < 1 or forecast_days > 15:
        raise ValueError("forecast_days must be between 1 and 15")
    
    logger.info(f"Generating {forecast_days}-day weather forecast for {city.name}")
    
    # Get initial conditions
    initial_conditions = get_initial_conditions(city)
    
    # Generate forecast
    forecast = predict_weather(
        initial_conditions=initial_conditions,
        forecast_days=forecast_days,
        latitude=city.latitude,
        longitude=city.longitude,
        model_path=model_path,
    )
    
    return forecast


def generate_climate_projection(
    city: City,
    scenario: Optional[Scenario] = None,
    projection_years: int = 10,
    model_path: Optional[str] = None,
) -> Dict:
    """
    Generate long-term climate projection using ClimaX (years).
    
    Args:
        city: City model instance
        scenario: Optional scenario with parameters
        projection_years: Number of years to project (1-50)
        model_path: Optional path to ClimaX model weights
    
    Returns:
        Dictionary with projection results
    """
    if projection_years < 1 or projection_years > 50:
        raise ValueError("projection_years must be between 1 and 50")
    
    logger.info(f"Generating {projection_years}-year climate projection for {city.name}")
    
    # Get historical data
    historical_data = get_historical_climate_data(city, years_back=5)
    
    # Get scenario parameters
    if scenario:
        scenario_parameters = scenario.parameters
    else:
        scenario_parameters = {}
    
    # Get urban features
    urban_features = get_urban_features(city)
    
    # Generate projection
    projection = predict_climate(
        historical_data=historical_data,
        scenario_parameters=scenario_parameters,
        projection_years=projection_years,
        latitude=city.latitude,
        longitude=city.longitude,
        urban_features=urban_features,
        model_path=model_path,
    )
    
    return projection


def save_weather_forecast_to_db(
    city: City,
    forecast: Dict,
    scenario: Optional[Scenario] = None,
) -> List[Prediction]:
    """
    Save weather forecast results to database.
    
    Args:
        city: City model instance
        forecast: Forecast dictionary from GraphCast
        scenario: Optional scenario
    
    Returns:
        List of created Prediction instances
    """
    predictions = []
    
    for forecast_point in forecast.get('forecast', []):
        timestamp = datetime.fromisoformat(forecast_point['timestamp'].replace('Z', '+00:00'))
        
        # Create or get scenario
        if not scenario:
            scenario, _ = Scenario.objects.get_or_create(
                name=f"Weather Forecast - {city.name}",
                city=city,
                defaults={
                    'description': f"Automatic weather forecast for {city.name}",
                    'time_horizon': '7d',
                }
            )
        
        # Create prediction
        prediction = Prediction.objects.create(
            scenario=scenario,
            model_type='weather',
            timestamp=timestamp,
            predictions={
                'temperature': forecast_point.get('temperature'),
                'humidity': forecast_point.get('humidity'),
                'precipitation': forecast_point.get('precipitation'),
                'wind_speed': forecast_point.get('wind_speed'),
                'wind_direction': forecast_point.get('wind_direction'),
                'pressure': forecast_point.get('pressure'),
            },
        )
        predictions.append(prediction)
    
    logger.info(f"Saved {len(predictions)} weather forecast predictions to database")
    return predictions


def save_climate_projection_to_db(
    city: City,
    projection: Dict,
    scenario: Scenario,
) -> List[Prediction]:
    """
    Save climate projection results to database.
    
    Args:
        city: City model instance
        projection: Projection dictionary from ClimaX
        scenario: Scenario instance
    
    Returns:
        List of created Prediction instances
    """
    predictions = []
    
    # Sample monthly data (not every single point to avoid database bloat)
    projection_points = projection.get('projection', [])
    # Take every 3rd point (quarterly) for long-term projections
    sampled_points = projection_points[::3] if len(projection_points) > 100 else projection_points
    
    for proj_point in sampled_points:
        timestamp = datetime.fromisoformat(proj_point['timestamp'].replace('Z', '+00:00'))
        
        # Create prediction
        prediction = Prediction.objects.create(
            scenario=scenario,
            model_type='climate',
            timestamp=timestamp,
            predictions={
                'temperature': proj_point.get('temperature'),
                'temperature_anomaly': proj_point.get('temperature_anomaly'),
                'humidity': proj_point.get('humidity'),
                'precipitation': proj_point.get('precipitation'),
                'precipitation_anomaly': proj_point.get('precipitation_anomaly'),
            },
        )
        predictions.append(prediction)
    
    logger.info(f"Saved {len(predictions)} climate projection predictions to database")
    return predictions

