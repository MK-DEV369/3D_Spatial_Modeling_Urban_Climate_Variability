from rest_framework import serializers
from rest_framework_gis.serializers import GeoFeatureModelSerializer
from core.models import City, Building, ClimateData, TrafficData, PollutionData, Scenario, Prediction


class CitySerializer(serializers.ModelSerializer):
    class Meta:
        model = City
        fields = ['id', 'name', 'country', 'latitude', 'longitude', 'bounds', 'metadata', 'created_at']


class BuildingSerializer(GeoFeatureModelSerializer):
    city_name = serializers.CharField(source='city.name', read_only=True)

    class Meta:
        model = Building
        geo_field = 'geometry'
        fields = ['id', 'osm_id', 'city', 'city_name', 'building_type', 'height', 'metadata']


class ClimateDataSerializer(serializers.ModelSerializer):
    city_name = serializers.CharField(source='city.name', read_only=True)

    class Meta:
        model = ClimateData
        fields = [
            'id', 'city', 'city_name', 'timestamp', 'temperature', 'humidity',
            'precipitation', 'wind_speed', 'wind_direction', 'pressure',
            'solar_radiation', 'metadata'
        ]


class TrafficDataSerializer(serializers.ModelSerializer):
    city_name = serializers.CharField(source='city.name', read_only=True)

    class Meta:
        model = TrafficData
        fields = [
            'id', 'city', 'city_name', 'timestamp', 'location', 'volume',
            'speed', 'congestion_level', 'metadata'
        ]


class PollutionDataSerializer(serializers.ModelSerializer):
    city_name = serializers.CharField(source='city.name', read_only=True)

    class Meta:
        model = PollutionData
        fields = [
            'id', 'city', 'city_name', 'timestamp', 'location', 'aqi',
            'pm25', 'pm10', 'no2', 'so2', 'co', 'o3', 'metadata'
        ]


class ScenarioSerializer(serializers.ModelSerializer):
    city_name = serializers.CharField(source='city.name', read_only=True)

    class Meta:
        model = Scenario
        fields = [
            'id', 'name', 'description', 'city', 'city_name', 'parameters',
            'time_horizon', 'created_at', 'updated_at'
        ]


class PredictionSerializer(serializers.ModelSerializer):
    scenario_name = serializers.CharField(source='scenario.name', read_only=True)

    class Meta:
        model = Prediction
        fields = [
            'id', 'scenario', 'scenario_name', 'model_type', 'timestamp',
            'predictions', 'created_at'
        ]

