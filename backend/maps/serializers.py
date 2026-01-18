"""
Serializers for OSM layer models (buildings, roads, water, green spaces).
"""
from rest_framework import serializers
from rest_framework_gis.serializers import GeoFeatureModelSerializer
from .models import BuildingsOSM, RoadsOSM, WaterOSM, GreenOSM


class BuildingsOSMSerializer(GeoFeatureModelSerializer):
    """Serializer for BuildingsOSM with GeoJSON support."""
    
    class Meta:
        model = BuildingsOSM
        geo_field = 'geom'
        fields = ['osm_id', 'name', 'building', 'geom', 'active', 'scenario_id', 'modified_at']
        read_only_fields = ['osm_id']  # OSM ID is primary key


class RoadsOSMSerializer(GeoFeatureModelSerializer):
    """Serializer for RoadsOSM with GeoJSON support."""
    
    class Meta:
        model = RoadsOSM
        geo_field = 'geom'
        fields = ['osm_id', 'name', 'highway', 'surface', 'layer', 'geom', 'active', 'scenario_id', 'modified_at']
        read_only_fields = ['osm_id']


class WaterOSMSerializer(GeoFeatureModelSerializer):
    """Serializer for WaterOSM with GeoJSON support."""
    
    class Meta:
        model = WaterOSM
        geo_field = 'geom'
        fields = ['osm_id', 'name', 'natural', 'water', 'geom', 'active', 'scenario_id', 'modified_at']
        read_only_fields = ['osm_id']


class GreenOSMSerializer(GeoFeatureModelSerializer):
    """Serializer for GreenOSM with GeoJSON support."""
    
    class Meta:
        model = GreenOSM
        geo_field = 'geom'
        fields = ['osm_id', 'name', 'leisure', 'landuse', 'natural', 'geom', 'active', 'scenario_id', 'modified_at']
        read_only_fields = ['osm_id']
