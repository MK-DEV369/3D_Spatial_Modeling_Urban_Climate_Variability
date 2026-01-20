"""
Runtime serializers for map rendering (Cesium / GeoJSON safe).

These serializers:
- Use simplified / transformed geometry (geom_simple)
- Expose only essential fields
- Avoid heavy metadata
- Are safe for large map rendering
"""

from rest_framework_gis.serializers import GeoFeatureModelSerializer
from .models import BuildingsOSM, RoadsOSM, WaterOSM, GreenOSM


class BuildingsRuntimeSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = BuildingsOSM
        geo_field = "geom"   # Annotated in queryset
        fields = (
            "osm_id",
            "active",
            "scenario_id",
        )


class RoadsRuntimeSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = RoadsOSM
        geo_field = "geom"
        fields = (
            "osm_id",
            "active",
            "scenario_id",
        )


class WaterRuntimeSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = WaterOSM
        geo_field = "geom"
        fields = (
            "osm_id",
            "active",
            "scenario_id",
        )


class GreenRuntimeSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = GreenOSM
        geo_field = "geom"
        fields = (
            "osm_id",
            "active",
            "scenario_id",
        )
