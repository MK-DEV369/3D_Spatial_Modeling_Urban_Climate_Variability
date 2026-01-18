from django.contrib import admin
from django.contrib.gis.admin import OSMGeoAdmin
from .models import City, ClimateData, TrafficData, PollutionData, Scenario, Prediction
# Note: Building model removed - use maps.models.BuildingsOSM instead (registered in maps/admin.py)


@admin.register(City)
class CityAdmin(OSMGeoAdmin):
    list_display = ('name', 'country', 'created_at')
    search_fields = ('name', 'country')

# BuildingAdmin removed - use maps.admin.BuildingsOSMAdmin instead


@admin.register(ClimateData)
class ClimateDataAdmin(admin.ModelAdmin):
    list_display = ('city', 'timestamp', 'temperature', 'humidity', 'precipitation')
    list_filter = ('city', 'timestamp')
    date_hierarchy = 'timestamp'


@admin.register(TrafficData)
class TrafficDataAdmin(admin.ModelAdmin):
    list_display = ('city', 'timestamp', 'volume', 'speed', 'congestion_level')
    list_filter = ('city', 'timestamp', 'congestion_level')
    date_hierarchy = 'timestamp'


@admin.register(PollutionData)
class PollutionDataAdmin(admin.ModelAdmin):
    list_display = ('city', 'timestamp', 'aqi', 'pm25', 'pm10')
    list_filter = ('city', 'timestamp')
    date_hierarchy = 'timestamp'


@admin.register(Scenario)
class ScenarioAdmin(admin.ModelAdmin):
    list_display = ('name', 'city', 'time_horizon', 'created_at')
    list_filter = ('city', 'time_horizon', 'created_at')
    search_fields = ('name', 'description')


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('scenario', 'model_type', 'timestamp', 'created_at')
    list_filter = ('scenario', 'model_type', 'created_at')
    date_hierarchy = 'created_at'

