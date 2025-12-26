"""
Celery tasks for async processing of climate, traffic, and pollution predictions.
"""
import logging
from celery import shared_task
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


@shared_task(bind=True, max_retries=3)
def generate_weather_forecast_task(self, city_id: int, forecast_days: int = 7, scenario_id: int = None):
    """
    Celery task to generate weather forecast asynchronously.
    
    Args:
        city_id: ID of the city
        forecast_days: Number of days to forecast (1-15)
        scenario_id: Optional scenario ID
    
    Returns:
        Dictionary with task results
    """
    try:
        city = City.objects.get(pk=city_id)
        scenario = Scenario.objects.get(pk=scenario_id) if scenario_id else None
        
        logger.info(f"Starting weather forecast task for {city.name}")
        
        # Generate forecast
        forecast = generate_weather_forecast(city, forecast_days=forecast_days)
        
        # Save to database
        predictions = save_weather_forecast_to_db(city, forecast, scenario=scenario)
        
        return {
            'status': 'success',
            'city_id': city_id,
            'forecast_days': forecast_days,
            'predictions_created': len(predictions),
        }
    
    except City.DoesNotExist:
        logger.error(f"City with ID {city_id} not found")
        raise
    except Exception as e:
        logger.error(f"Error in weather forecast task: {str(e)}")
        raise self.retry(exc=e, countdown=60)


@shared_task(bind=True, max_retries=3)
def generate_climate_projection_task(
    self,
    city_id: int,
    scenario_id: int,
    projection_years: int = 10,
):
    """
    Celery task to generate climate projection asynchronously.
    
    Args:
        city_id: ID of the city
        scenario_id: Scenario ID
        projection_years: Number of years to project (1-50)
    
    Returns:
        Dictionary with task results
    """
    try:
        city = City.objects.get(pk=city_id)
        scenario = Scenario.objects.get(pk=scenario_id)
        
        logger.info(f"Starting climate projection task for {city.name}, scenario: {scenario.name}")
        
        # Generate projection
        projection = generate_climate_projection(
            city,
            scenario=scenario,
            projection_years=projection_years,
        )
        
        # Save to database
        predictions = save_climate_projection_to_db(city, projection, scenario)
        
        return {
            'status': 'success',
            'city_id': city_id,
            'scenario_id': scenario_id,
            'projection_years': projection_years,
            'predictions_created': len(predictions),
        }
    
    except (City.DoesNotExist, Scenario.DoesNotExist) as e:
        logger.error(f"City or Scenario not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in climate projection task: {str(e)}")
        raise self.retry(exc=e, countdown=60)


@shared_task(bind=True, max_retries=3)
def generate_traffic_prediction_task(
    self,
    city_id: int,
    prediction_hours: int = 24,
    scenario_id: int = None,
):
    """
    Celery task to generate traffic prediction asynchronously.
    
    Args:
        city_id: ID of the city
        prediction_hours: Number of hours to predict (1-168)
        scenario_id: Optional scenario ID
    
    Returns:
        Dictionary with task results
    """
    try:
        city = City.objects.get(pk=city_id)
        scenario = Scenario.objects.get(pk=scenario_id) if scenario_id else None
        
        logger.info(f"Starting traffic prediction task for {city.name}")
        
        # Generate prediction
        prediction = generate_traffic_prediction(
            city,
            prediction_hours=prediction_hours,
            scenario=scenario,
        )
        
        # Save to database
        predictions = save_traffic_prediction_to_db(city, prediction, scenario=scenario)
        
        return {
            'status': 'success',
            'city_id': city_id,
            'prediction_hours': prediction_hours,
            'predictions_created': len(predictions),
        }
    
    except City.DoesNotExist:
        logger.error(f"City with ID {city_id} not found")
        raise
    except Exception as e:
        logger.error(f"Error in traffic prediction task: {str(e)}")
        raise self.retry(exc=e, countdown=60)


@shared_task(bind=True, max_retries=3)
def generate_pollution_prediction_task(
    self,
    city_id: int,
    prediction_hours: int = 24,
    scenario_id: int = None,
):
    """
    Celery task to generate pollution prediction asynchronously.
    
    Args:
        city_id: ID of the city
        prediction_hours: Number of hours to predict (1-168)
        scenario_id: Optional scenario ID
    
    Returns:
        Dictionary with task results
    """
    try:
        city = City.objects.get(pk=city_id)
        scenario = Scenario.objects.get(pk=scenario_id) if scenario_id else None
        
        logger.info(f"Starting pollution prediction task for {city.name}")
        
        # Generate prediction
        prediction = generate_pollution_prediction(
            city,
            prediction_hours=prediction_hours,
            scenario=scenario,
        )
        
        # Save to database
        predictions = save_pollution_prediction_to_db(city, prediction, scenario=scenario)
        
        return {
            'status': 'success',
            'city_id': city_id,
            'prediction_hours': prediction_hours,
            'predictions_created': len(predictions),
        }
    
    except City.DoesNotExist:
        logger.error(f"City with ID {city_id} not found")
        raise
    except Exception as e:
        logger.error(f"Error in pollution prediction task: {str(e)}")
        raise self.retry(exc=e, countdown=60)

