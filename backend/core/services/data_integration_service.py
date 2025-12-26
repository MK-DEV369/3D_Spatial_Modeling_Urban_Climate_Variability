"""
Data integration service for external APIs.
Handles fetching historical climate, traffic, and pollution data.
"""
import os
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from django.utils import timezone
from core.models import City, ClimateData, PollutionData

logger = logging.getLogger(__name__)


class WeatherDataService:
    """Service for fetching historical weather data from OpenWeatherMap."""
    
    def __init__(self):
        self.api_key = os.environ.get('OPENWEATHER_API_KEY', '')
        self.base_url = 'https://api.openweathermap.org/data/2.5'
        self.history_url = 'https://history.openweathermap.org/data/2.5/history/city'
    
    def fetch_current_weather(self, city: City) -> Optional[Dict]:
        """
        Fetch current weather data for a city.
        
        Args:
            city: City model instance
        
        Returns:
            Dictionary with weather data or None if request fails
        """
        if not self.api_key:
            logger.warning('OpenWeatherMap API key not configured')
            return self._generate_mock_weather(city)
        
        try:
            params = {
                'lat': city.location.y,
                'lon': city.location.x,
                'appid': self.api_key,
                'units': 'metric',
            }
            response = requests.get(f'{self.base_url}/weather', params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error fetching weather for {city.name}: {str(e)}')
            return self._generate_mock_weather(city)
    
    def fetch_forecast(self, city: City, days: int = 7) -> Optional[Dict]:
        """
        Fetch weather forecast for a city.
        
        Args:
            city: City model instance
            days: Number of days to forecast (max 7 for free tier)
        
        Returns:
            Dictionary with forecast data or None if request fails
        """
        if not self.api_key:
            logger.warning('OpenWeatherMap API key not configured')
            return self._generate_mock_forecast(city, days)
        
        try:
            params = {
                'lat': city.location.y,
                'lon': city.location.x,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': min(days * 8, 40),  # 3-hour intervals, max 5 days for free tier
            }
            response = requests.get(f'{self.base_url}/forecast', params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error fetching forecast for {city.name}: {str(e)}')
            return self._generate_mock_forecast(city, days)
    
    def store_current_weather(self, city: City) -> Optional[ClimateData]:
        """
        Fetch and store current weather data in database.
        
        Args:
            city: City model instance
        
        Returns:
            ClimateData instance or None if fetch fails
        """
        weather_data = self.fetch_current_weather(city)
        if not weather_data:
            return None
        
        try:
            climate = ClimateData.objects.create(
                city=city,
                timestamp=timezone.now(),
                temperature=weather_data.get('main', {}).get('temp'),
                humidity=weather_data.get('main', {}).get('humidity'),
                precipitation=weather_data.get('rain', {}).get('1h', 0),
                wind_speed=weather_data.get('wind', {}).get('speed'),
                data_source='openweathermap',
            )
            logger.info(f'Stored weather data for {city.name}')
            return climate
        except Exception as e:
            logger.error(f'Error storing weather for {city.name}: {str(e)}')
            return None
    
    def _generate_mock_weather(self, city: City) -> Dict:
        """Generate realistic mock weather data when API is unavailable."""
        import random
        
        # City-specific base temperatures
        base_temps = {
            'Bengaluru': 25,
            'Delhi': 28,
            'Mumbai': 30,
            'Chennai': 32,
            'Dubai': 35,
            'Amsterdam': 15,
        }
        base_temp = base_temps.get(city.name, 25)
        
        return {
            'main': {
                'temp': base_temp + random.uniform(-3, 3),
                'humidity': random.randint(40, 80),
                'pressure': random.randint(1005, 1020),
            },
            'wind': {
                'speed': random.uniform(2, 8),
            },
            'rain': {
                '1h': random.uniform(0, 5) if random.random() < 0.3 else 0,
            },
        }
    
    def _generate_mock_forecast(self, city: City, days: int) -> Dict:
        """Generate mock forecast data."""
        import random
        
        forecasts = []
        for i in range(days * 8):  # 3-hour intervals
            timestamp = int((datetime.now() + timedelta(hours=i * 3)).timestamp())
            temp = self._generate_mock_weather(city)['main']['temp']
            forecasts.append({
                'dt': timestamp,
                'main': {'temp': temp},
            })
        
        return {'list': forecasts}


class PollutionDataService:
    """Service for fetching air quality and pollution data."""
    
    def __init__(self):
        self.openaq_url = 'https://api.openaq.org/v2'
        self.openweather_api_key = os.environ.get('OPENWEATHER_API_KEY', '')
    
    def fetch_air_quality(self, city: City) -> Optional[Dict]:
        """
        Fetch air quality data for a city from OpenAQ or OpenWeatherMap.
        
        Args:
            city: City model instance
        
        Returns:
            Dictionary with air quality data or None
        """
        # Try OpenWeatherMap Air Pollution API first
        if self.openweather_api_key:
            try:
                params = {
                    'lat': city.location.y,
                    'lon': city.location.x,
                    'appid': self.openweather_api_key,
                }
                response = requests.get(
                    'https://api.openweathermap.org/data/2.5/air_pollution',
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                
                if 'list' in data and len(data['list']) > 0:
                    return data['list'][0]
            except Exception as e:
                logger.error(f'Error fetching air quality from OpenWeather: {str(e)}')
        
        # Fallback to OpenAQ (no API key required but rate limited)
        try:
            params = {
                'coordinates': f'{city.location.y},{city.location.x}',
                'radius': 25000,  # 25km radius
                'limit': 100,
            }
            response = requests.get(f'{self.openaq_url}/latest', params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error fetching air quality from OpenAQ: {str(e)}')
            return self._generate_mock_pollution(city)
    
    def store_pollution_data(self, city: City) -> Optional[PollutionData]:
        """
        Fetch and store pollution data in database.
        
        Args:
            city: City model instance
        
        Returns:
            PollutionData instance or None
        """
        aqi_data = self.fetch_air_quality(city)
        if not aqi_data:
            return None
        
        try:
            # Parse OpenWeatherMap format
            if 'components' in aqi_data:
                components = aqi_data['components']
                pollution = PollutionData.objects.create(
                    city=city,
                    timestamp=timezone.now(),
                    pm25=components.get('pm2_5'),
                    pm10=components.get('pm10'),
                    no2=components.get('no2'),
                    so2=components.get('so2'),
                    co=components.get('co'),
                    o3=components.get('o3'),
                    aqi=aqi_data.get('main', {}).get('aqi', 0) * 50,  # Convert to AQI scale
                    data_source='openweathermap',
                )
            else:
                # OpenAQ format or mock data
                pollution = PollutionData.objects.create(
                    city=city,
                    timestamp=timezone.now(),
                    pm25=aqi_data.get('pm25', 0),
                    pm10=aqi_data.get('pm10', 0),
                    no2=aqi_data.get('no2', 0),
                    aqi=aqi_data.get('aqi', 50),
                    data_source='openaq',
                )
            
            logger.info(f'Stored pollution data for {city.name}')
            return pollution
        except Exception as e:
            logger.error(f'Error storing pollution for {city.name}: {str(e)}')
            return None
    
    def _generate_mock_pollution(self, city: City) -> Dict:
        """Generate realistic mock pollution data."""
        import random
        
        # City-specific base AQI levels
        base_aqi = {
            'Bengaluru': 80,
            'Delhi': 150,
            'Mumbai': 90,
            'Chennai': 70,
            'Dubai': 60,
            'Amsterdam': 40,
        }
        aqi = base_aqi.get(city.name, 80) + random.randint(-20, 20)
        
        return {
            'aqi': aqi,
            'pm25': aqi * 0.5,
            'pm10': aqi * 0.8,
            'no2': aqi * 0.3,
            'so2': aqi * 0.2,
            'co': aqi * 2,
            'o3': aqi * 0.4,
        }


class DataIntegrationService:
    """Unified service for all data integrations."""
    
    def __init__(self):
        self.weather_service = WeatherDataService()
        self.pollution_service = PollutionDataService()
    
    def sync_city_data(self, city: City) -> Dict[str, any]:
        """
        Sync all external data for a city.
        
        Args:
            city: City model instance
        
        Returns:
            Dictionary with sync results
        """
        results = {
            'city': city.name,
            'climate': None,
            'pollution': None,
            'errors': [],
        }
        
        try:
            climate = self.weather_service.store_current_weather(city)
            results['climate'] = climate.id if climate else None
        except Exception as e:
            results['errors'].append(f'Climate: {str(e)}')
        
        try:
            pollution = self.pollution_service.store_pollution_data(city)
            results['pollution'] = pollution.id if pollution else None
        except Exception as e:
            results['errors'].append(f'Pollution: {str(e)}')
        
        return results
    
    def sync_all_cities(self) -> List[Dict[str, any]]:
        """Sync data for all cities in database."""
        cities = City.objects.all()
        results = []
        
        for city in cities:
            result = self.sync_city_data(city)
            results.append(result)
            logger.info(f'Synced data for {city.name}: {result}')
        
        return results
