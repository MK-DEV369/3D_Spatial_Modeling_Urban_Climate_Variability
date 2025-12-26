"""
Traffic prediction service that orchestrates traffic modeling.

This service provides traffic predictions for urban areas considering
time patterns, weather, events, and historical data.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from django.utils import timezone
from core.models import City, TrafficData, ClimateData, Scenario, Prediction
from ml_pipeline.traffic.inference import predict_traffic, get_traffic_instance

logger = logging.getLogger(__name__)


def get_historical_traffic_data(city: City, hours_back: int = 168) -> List[Dict]:
    """
    Get historical traffic data for model input.
    
    Args:
        city: City model instance
        hours_back: Number of hours to look back (default: 1 week)
    
    Returns:
        List of traffic data dictionaries
    """
    cutoff_date = timezone.now() - timedelta(hours=hours_back)
    historical = TrafficData.objects.filter(
        city=city,
        timestamp__gte=cutoff_date
    ).order_by('timestamp')
    
    data = []
    for record in historical:
        data.append({
            'timestamp': record.timestamp.isoformat(),
            'volume': record.volume,
            'speed': record.speed,
            'congestion_level': record.congestion_level,
        })
    
    return data


def get_weather_for_traffic(city: City, prediction_hours: int = 24) -> Optional[Dict]:
    """
    Get weather data for traffic prediction.
    
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
            'precipitation': recent_weather.precipitation or 0.0,
            'wind_speed': recent_weather.wind_speed or 0.0,
        }
    
    return None


def get_time_features(timestamp: datetime) -> Dict:
    """
    Extract time-based features for traffic prediction.
    
    Args:
        timestamp: Datetime to extract features from
    
    Returns:
        Dictionary with time features
    """
    return {
        'hour_of_day': timestamp.hour,
        'day_of_week': timestamp.weekday(),  # 0=Monday, 6=Sunday
        'is_weekend': timestamp.weekday() >= 5,
        'is_holiday': False,  # TODO: Implement holiday detection
        'month': timestamp.month,
    }


def generate_traffic_prediction(
    city: City,
    prediction_hours: int = 24,
    scenario: Optional[Scenario] = None,
    model_path: Optional[str] = None,
) -> Dict:
    """
    Generate traffic prediction for a city.
    
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
    
    logger.info(f"Generating {prediction_hours}-hour traffic prediction for {city.name}")
    
    # Get historical traffic data
    historical_data = get_historical_traffic_data(city, hours_back=min(168, prediction_hours * 2))
    
    # Get weather data
    weather_data = get_weather_for_traffic(city, prediction_hours)
    
    # Get time features for first prediction timestamp
    current_time = timezone.now()
    time_features = get_time_features(current_time)
    
    # Get events from scenario if available
    events = None
    if scenario and scenario.parameters.get('events'):
        events = scenario.parameters.get('events')
    
    # Generate prediction
    prediction = predict_traffic(
        historical_data=historical_data,
        prediction_hours=prediction_hours,
        time_features=time_features,
        weather_data=weather_data,
        events=events,
        model_path=model_path,
    )
    
    return prediction


def save_traffic_prediction_to_db(
    city: City,
    prediction: Dict,
    scenario: Optional[Scenario] = None,
) -> List[Prediction]:
    """
    Save traffic prediction results to database.
    
    Args:
        city: City model instance
        prediction: Prediction dictionary from traffic model
        scenario: Optional scenario
    
    Returns:
        List of created Prediction instances
    """
    predictions = []
    
    # Create or get scenario
    if not scenario:
        scenario, _ = Scenario.objects.get_or_create(
            name=f"Traffic Prediction - {city.name}",
            city=city,
            defaults={
                'description': f"Automatic traffic prediction for {city.name}",
                'time_horizon': '1d',
            }
        )
    
    # Save predictions (sample hourly data)
    for pred_point in prediction.get('predictions', []):
        timestamp = datetime.fromisoformat(pred_point['timestamp'].replace('Z', '+00:00'))
        
        # Create prediction record
        prediction_obj = Prediction.objects.create(
            scenario=scenario,
            model_type='traffic',
            timestamp=timestamp,
            predictions={
                'volume': pred_point.get('volume'),
                'speed': pred_point.get('speed'),
                'congestion_level': pred_point.get('congestion_level'),
            },
        )
        predictions.append(prediction_obj)
    
    logger.info(f"Saved {len(predictions)} traffic predictions to database")
    return predictions

