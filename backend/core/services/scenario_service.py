"""
Scenario simulation service that orchestrates all ML models.

This service provides end-to-end scenario simulation by running climate,
traffic, and pollution predictions together based on user-defined scenarios.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from django.utils import timezone
from core.models import City, Scenario, Prediction
from core.services.climate_service import (
    generate_weather_forecast,
    generate_climate_projection,
    save_weather_forecast_to_db,
    save_climate_projection_to_db,
)
from core.services.traffic_service import (
    generate_traffic_prediction,
    save_traffic_prediction_to_db,
)
from core.services.pollution_service import (
    generate_pollution_prediction,
    save_pollution_prediction_to_db,
)

logger = logging.getLogger(__name__)


def get_prediction_horizon_hours(time_horizon: str) -> int:
    """
    Convert time horizon string to hours.
    
    Args:
        time_horizon: Time horizon string (1d, 7d, 30d, 1y, 5y, 10y)
    
    Returns:
        Number of hours
    """
    horizon_map = {
        '1d': 24,
        '7d': 168,  # 7 * 24
        '30d': 720,  # 30 * 24 (but capped at 168 for hourly predictions)
        '1y': 8760,  # 365 * 24 (but capped at 168 for hourly predictions)
        '5y': 43800,  # 5 * 365 * 24 (but capped at 168 for hourly predictions)
        '10y': 87600,  # 10 * 365 * 24 (but capped at 168 for hourly predictions)
    }
    hours = horizon_map.get(time_horizon, 24)
    # Cap at 168 hours (1 week) for hourly predictions
    # Longer horizons use monthly/yearly projections
    return min(hours, 168)


def get_projection_years(time_horizon: str) -> int:
    """
    Convert time horizon string to years for long-term projections.
    
    Args:
        time_horizon: Time horizon string (1d, 7d, 30d, 1y, 5y, 10y)
    
    Returns:
        Number of years (0 for short-term horizons)
    """
    horizon_map = {
        '1d': 0,
        '7d': 0,
        '30d': 0,
        '1y': 1,
        '5y': 5,
        '10y': 10,
    }
    return horizon_map.get(time_horizon, 0)


def run_scenario_simulation(
    scenario: Scenario,
    include_models: Optional[List[str]] = None,
    async_mode: bool = False,
) -> Dict:
    """
    Run complete scenario simulation with all ML models.
    
    Args:
        scenario: Scenario instance
        include_models: List of models to run ['climate', 'traffic', 'pollution']
                       If None, runs all models
        async_mode: If True, returns task IDs instead of running synchronously
    
    Returns:
        Dictionary with simulation results or task IDs
    """
    if include_models is None:
        include_models = ['climate', 'traffic', 'pollution']
    
    city = scenario.city
    time_horizon = scenario.time_horizon
    
    logger.info(f"Running scenario simulation for {scenario.name} (city: {city.name}, horizon: {time_horizon})")
    
    results = {
        'scenario_id': scenario.id,
        'scenario_name': scenario.name,
        'city_id': city.id,
        'city_name': city.name,
        'time_horizon': time_horizon,
        'models_run': [],
        'predictions_created': {},
        'errors': [],
    }
    
    # Determine prediction parameters based on time horizon
    projection_years = get_projection_years(time_horizon)
    prediction_hours = get_prediction_horizon_hours(time_horizon)
    
    # Run climate predictions
    if 'climate' in include_models:
        try:
            if projection_years > 0:
                # Long-term climate projection
                logger.info(f"Generating {projection_years}-year climate projection")
                projection = generate_climate_projection(
                    city,
                    scenario=scenario,
                    projection_years=projection_years,
                )
                predictions = save_climate_projection_to_db(city, projection, scenario)
                results['models_run'].append('climate_projection')
                results['predictions_created']['climate'] = len(predictions)
            else:
                # Short-term weather forecast
                logger.info(f"Generating {prediction_hours // 24}-day weather forecast")
                forecast = generate_weather_forecast(
                    city,
                    forecast_days=prediction_hours // 24,
                )
                predictions = save_weather_forecast_to_db(city, forecast, scenario=scenario)
                results['models_run'].append('weather_forecast')
                results['predictions_created']['weather'] = len(predictions)
        except Exception as e:
            logger.error(f"Error running climate prediction: {str(e)}")
            results['errors'].append({'model': 'climate', 'error': str(e)})
    
    # Run traffic predictions (only for short-term horizons)
    if 'traffic' in include_models and projection_years == 0:
        try:
            logger.info(f"Generating {prediction_hours}-hour traffic prediction")
            traffic_prediction = generate_traffic_prediction(
                city,
                prediction_hours=prediction_hours,
                scenario=scenario,
            )
            predictions = save_traffic_prediction_to_db(city, traffic_prediction, scenario=scenario)
            results['models_run'].append('traffic')
            results['predictions_created']['traffic'] = len(predictions)
        except Exception as e:
            logger.error(f"Error running traffic prediction: {str(e)}")
            results['errors'].append({'model': 'traffic', 'error': str(e)})
    
    # Run pollution predictions (only for short-term horizons)
    if 'pollution' in include_models and projection_years == 0:
        try:
            logger.info(f"Generating {prediction_hours}-hour pollution prediction")
            pollution_prediction = generate_pollution_prediction(
                city,
                prediction_hours=prediction_hours,
                scenario=scenario,
            )
            predictions = save_pollution_prediction_to_db(city, pollution_prediction, scenario=scenario)
            results['models_run'].append('pollution')
            results['predictions_created']['pollution'] = len(predictions)
        except Exception as e:
            logger.error(f"Error running pollution prediction: {str(e)}")
            results['errors'].append({'model': 'pollution', 'error': str(e)})
    
    results['status'] = 'success' if not results['errors'] else 'partial_success'
    logger.info(f"Scenario simulation completed: {results['status']}")
    
    return results


def get_scenario_summary(scenario: Scenario) -> Dict:
    """
    Get summary of all predictions for a scenario.
    
    Args:
        scenario: Scenario instance
    
    Returns:
        Dictionary with scenario summary
    """
    predictions = Prediction.objects.filter(scenario=scenario).order_by('timestamp')
    
    summary = {
        'scenario_id': scenario.id,
        'scenario_name': scenario.name,
        'city_name': scenario.city.name,
        'time_horizon': scenario.time_horizon,
        'total_predictions': predictions.count(),
        'predictions_by_model': {},
        'date_range': {},
    }
    
    # Group by model type
    for model_type in ['weather', 'climate', 'traffic', 'pollution']:
        model_predictions = predictions.filter(model_type=model_type)
        count = model_predictions.count()
        if count > 0:
            summary['predictions_by_model'][model_type] = {
                'count': count,
                'first_timestamp': model_predictions.first().timestamp.isoformat() if model_predictions.exists() else None,
                'last_timestamp': model_predictions.last().timestamp.isoformat() if model_predictions.exists() else None,
            }
    
    # Overall date range
    if predictions.exists():
        summary['date_range'] = {
            'start': predictions.first().timestamp.isoformat(),
            'end': predictions.last().timestamp.isoformat(),
        }
    
    return summary

