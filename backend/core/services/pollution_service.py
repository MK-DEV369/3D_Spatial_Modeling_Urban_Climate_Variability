"""
Pollution prediction service that orchestrates pollution modeling.

This service provides air quality and pollution predictions for urban areas
considering traffic, weather, urban features, and historical data.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from django.utils import timezone
from core.models import City, PollutionData, TrafficData, ClimateData, Scenario, Prediction
from ml_pipeline.pollution.inference import predict_pollution, get_pollution_instance

logger = logging.getLogger(__name__)


def get_historical_pollution_data(city: City, hours_back: int = 168) -> List[Dict]:
    """
    Get historical pollution data for model input.
    
    Args:
        city: City model instance
        hours_back: Number of hours to look back (default: 1 week)
    
    Returns:
        List of pollution data dictionaries
    """
    cutoff_date = timezone.now() - timedelta(hours=hours_back)
    historical = PollutionData.objects.filter(
        city=city,
        timestamp__gte=cutoff_date
    ).order_by('timestamp')
    
    data = []
    for record in historical:
        data.append({
            'timestamp': record.timestamp.isoformat(),
            'aqi': record.aqi,
            'pm25': record.pm25,
            'pm10': record.pm10,
            'no2': record.no2,
            'so2': record.so2,
            'co': record.co,
            'o3': record.o3,
        })
    
    return data


def get_traffic_for_pollution(city: City, prediction_hours: int = 24) -> Optional[List[Dict]]:
    """
    Get traffic data for pollution prediction.
    
    Args:
        city: City model instance
        prediction_hours: Number of hours to get traffic for
    
    Returns:
        List of traffic data dictionaries or None
    """
    # Get recent traffic data
    cutoff_date = timezone.now() - timedelta(hours=prediction_hours)
    traffic_data = TrafficData.objects.filter(
        city=city,
        timestamp__gte=cutoff_date
    ).order_by('timestamp')
    
    if traffic_data.exists():
        data = []
        for record in traffic_data:
            data.append({
                'timestamp': record.timestamp.isoformat(),
                'volume': record.volume,
                'speed': record.speed,
            })
        return data
    
    return None


def get_weather_for_pollution(city: City, prediction_hours: int = 24) -> Optional[Dict]:
    """
    Get weather data for pollution prediction.
    
    Args:
        city: City model instance
        prediction_hours: Number of hours to get weather for
    
    Returns:
        Dictionary with weather data or None
    """
    # Get recent weather data
    recent_weather = ClimateData.objects.filter(
        city=city
    ).order_by('-timestamp').first()
    
    if recent_weather:
        return {
            'temperature': recent_weather.temperature,
            'humidity': recent_weather.humidity or 60.0,
            'precipitation': recent_weather.precipitation or 0.0,
            'wind_speed': recent_weather.wind_speed or 5.0,
        }
    
    return None


def get_urban_features(city: City) -> Dict:
    """
    Extract urban features from city data for pollution modeling.
    
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
    
    # Get vegetation coverage from metadata
    vegetation_coverage = city.metadata.get('vegetation_coverage', 20.0)  # Percentage
    
    # Industrial activity (placeholder - would come from external data)
    industrial_activity = city.metadata.get('industrial_activity', 0.5)  # 0-1 scale
    
    return {
        'building_density': building_density,
        'vegetation_coverage': vegetation_coverage,
        'industrial_activity': industrial_activity,
    }


def generate_pollution_prediction(
    city: City,
    prediction_hours: int = 24,
    scenario: Optional[Scenario] = None,
    model_path: Optional[str] = None,
) -> Dict:
    """
    Generate pollution prediction for a city.
    
    Args:
        city: City model instance
        prediction_hours: Number of hours to predict (1-168)
        scenario: Optional scenario with parameters
        model_path: Optional path to model weights
    
    Returns:
        Dictionary with prediction results
    """
    if prediction_hours < 1 or prediction_hours > 168:
        raise ValueError("prediction_hours must be between 1 and 168 (1 week)")
    
    logger.info(f"Generating {prediction_hours}-hour pollution prediction for {city.name}")
    
    # Get historical pollution data
    historical_data = get_historical_pollution_data(city, hours_back=min(168, prediction_hours * 2))
    
    # Get traffic data
    traffic_data = get_traffic_for_pollution(city, prediction_hours)
    
    # Get weather data
    weather_data = get_weather_for_pollution(city, prediction_hours)
    
    # Get urban features
    urban_features = get_urban_features(city)
    
    # Apply scenario parameters if available
    if scenario and scenario.parameters:
        # Update urban features based on scenario
        if 'vegetation_change' in scenario.parameters:
            vegetation_change = scenario.parameters['vegetation_change']
            urban_features['vegetation_coverage'] = max(0, min(100, 
                urban_features['vegetation_coverage'] + vegetation_change))
        
        if 'building_density_change' in scenario.parameters:
            building_density_change = scenario.parameters['building_density_change']
            urban_features['building_density'] = max(0,
                urban_features['building_density'] * (1 + building_density_change / 100))
    
    # Generate prediction
    prediction = predict_pollution(
        historical_data=historical_data,
        prediction_hours=prediction_hours,
        traffic_data=traffic_data,
        weather_data=weather_data,
        urban_features=urban_features,
        model_path=model_path,
    )
    
    return prediction


def save_pollution_prediction_to_db(
    city: City,
    prediction: Dict,
    scenario: Optional[Scenario] = None,
) -> List[Prediction]:
    """
    Save pollution prediction results to database.
    
    Args:
        city: City model instance
        prediction: Prediction dictionary from pollution model
        scenario: Optional scenario
    
    Returns:
        List of created Prediction instances
    """
    predictions = []
    
    # Create or get scenario
    if not scenario:
        scenario, _ = Scenario.objects.get_or_create(
            name=f"Pollution Prediction - {city.name}",
            city=city,
            defaults={
                'description': f"Automatic pollution prediction for {city.name}",
                'time_horizon': '1d',
            }
        )
    
    # Save predictions (sample hourly data)
    for pred_point in prediction.get('predictions', []):
        timestamp = datetime.fromisoformat(pred_point['timestamp'].replace('Z', '+00:00'))
        
        # Create prediction record
        prediction_obj = Prediction.objects.create(
            scenario=scenario,
            model_type='pollution',
            timestamp=timestamp,
            predictions={
                'aqi': pred_point.get('aqi'),
                'pm25': pred_point.get('pm25'),
                'pm10': pred_point.get('pm10'),
                'no2': pred_point.get('no2'),
                'so2': pred_point.get('so2'),
                'co': pred_point.get('co'),
                'o3': pred_point.get('o3'),
            },
        )
        predictions.append(prediction_obj)
    
    logger.info(f"Saved {len(predictions)} pollution predictions to database")
    return predictions

