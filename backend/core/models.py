from django.contrib.gis.db import models
from django.contrib.postgres.fields import JSONField
from django.utils import timezone


class City(models.Model):
    """
    Model representing a city for climate modeling.
    """
    name = models.CharField(max_length=200)
    country = models.CharField(max_length=100, default='India')
    latitude = models.FloatField()
    longitude = models.FloatField()
    bounds = models.PolygonField(help_text="City boundary polygon", null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True, help_text="Additional city metadata")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Cities"
        ordering = ['name']

    def __str__(self):
        return f"{self.name}, {self.country}"


class ClimateData(models.Model):
    """
    Model for storing historical and predicted climate data.
    """
    city = models.ForeignKey(City, on_delete=models.CASCADE, related_name='climate_data')
    timestamp = models.DateTimeField()
    temperature = models.FloatField(help_text="Temperature in Celsius")
    humidity = models.FloatField(null=True, blank=True, help_text="Humidity percentage")
    precipitation = models.FloatField(null=True, blank=True, help_text="Precipitation in mm")
    wind_speed = models.FloatField(null=True, blank=True, help_text="Wind speed in m/s")
    wind_direction = models.FloatField(null=True, blank=True, help_text="Wind direction in degrees")
    pressure = models.FloatField(null=True, blank=True, help_text="Atmospheric pressure in hPa")
    solar_radiation = models.FloatField(null=True, blank=True, help_text="Solar radiation in W/m²")
    metadata = models.JSONField(default=dict, blank=True, help_text="Additional climate metadata")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['city', 'timestamp']),
            models.Index(fields=['timestamp']),
        ]

    def __str__(self):
        return f"Climate data for {self.city.name} at {self.timestamp}"


class TrafficData(models.Model):
    """
    Model for storing historical and predicted traffic data.
    """
    CONGESTION_LEVELS = [
        ('low', 'Low'),
        ('moderate', 'Moderate'),
        ('high', 'High'),
        ('severe', 'Severe'),
    ]

    city = models.ForeignKey(City, on_delete=models.CASCADE, related_name='traffic_data')
    timestamp = models.DateTimeField()
    location = models.PointField(null=True, blank=True, help_text="Traffic measurement location")
    volume = models.IntegerField(null=True, blank=True, help_text="Traffic volume (vehicles/hour)")
    speed = models.FloatField(null=True, blank=True, help_text="Average speed in km/h")
    congestion_level = models.CharField(max_length=20, choices=CONGESTION_LEVELS, blank=True)
    metadata = models.JSONField(default=dict, blank=True, help_text="Additional traffic metadata")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['city', 'timestamp']),
            models.Index(fields=['timestamp']),
        ]

    def __str__(self):
        return f"Traffic data for {self.city.name} at {self.timestamp}"


class PollutionData(models.Model):
    """
    Model for storing historical and predicted pollution/air quality data.
    """
    city = models.ForeignKey(City, on_delete=models.CASCADE, related_name='pollution_data')
    timestamp = models.DateTimeField()
    location = models.PointField(null=True, blank=True, help_text="Pollution measurement location")
    aqi = models.IntegerField(null=True, blank=True, help_text="Air Quality Index")
    pm25 = models.FloatField(null=True, blank=True, help_text="PM2.5 concentration in µg/m³")
    pm10 = models.FloatField(null=True, blank=True, help_text="PM10 concentration in µg/m³")
    no2 = models.FloatField(null=True, blank=True, help_text="NO2 concentration in µg/m³")
    so2 = models.FloatField(null=True, blank=True, help_text="SO2 concentration in µg/m³")
    co = models.FloatField(null=True, blank=True, help_text="CO concentration in mg/m³")
    o3 = models.FloatField(null=True, blank=True, help_text="O3 concentration in µg/m³")
    metadata = models.JSONField(default=dict, blank=True, help_text="Additional pollution metadata")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['city', 'timestamp']),
            models.Index(fields=['timestamp']),
        ]

    def __str__(self):
        return f"Pollution data for {self.city.name} at {self.timestamp}"


class Scenario(models.Model):
    """
    Model for storing user-defined climate scenarios.
    """
    TIME_HORIZON_CHOICES = [
        ('1d', '1 Day'),
        ('7d', '7 Days'),
        ('30d', '30 Days'),
        ('1y', '1 Year'),
        ('5y', '5 Years'),
        ('10y', '10 Years'),
    ]

    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    city = models.ForeignKey(City, on_delete=models.CASCADE, related_name='scenarios')
    parameters = models.JSONField(default=dict, help_text="Scenario parameters (e.g., vegetation_change, building_density)")
    time_horizon = models.CharField(max_length=10, choices=TIME_HORIZON_CHOICES, default='1y')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.city.name})"


class Prediction(models.Model):
    """
    Model for storing ML model predictions.
    """
    MODEL_TYPES = [
        ('weather', 'Weather (GraphCast)'),
        ('climate', 'Climate (ClimaX)'),
        ('traffic', 'Traffic'),
        ('pollution', 'Pollution'),
    ]

    scenario = models.ForeignKey(Scenario, on_delete=models.CASCADE, related_name='predictions')
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    timestamp = models.DateTimeField(help_text="Timestamp for which prediction is made")
    predictions = models.JSONField(help_text="Prediction results (format depends on model_type)")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp', '-created_at']
        indexes = [
            models.Index(fields=['scenario', 'model_type']),
            models.Index(fields=['timestamp']),
        ]

    def __str__(self):
        return f"{self.model_type} prediction for {self.scenario.name} at {self.timestamp}"

