from django.contrib import admin
from django.contrib.gis import admin as gis_admin
from .models import BuildingsOSM, RoadsOSM, WaterOSM, GreenOSM


@admin.register(BuildingsOSM)
class BuildingsOSMAdmin(gis_admin.OSMGeoAdmin):
    list_display = ['osm_id', 'name', 'building', 'active', 'scenario_id', 'modified_at']
    list_filter = ['active', 'scenario_id', 'building']
    search_fields = ['osm_id', 'name']
    readonly_fields = ['osm_id', 'modified_at']


@admin.register(RoadsOSM)
class RoadsOSMAdmin(gis_admin.OSMGeoAdmin):
    list_display = ['osm_id', 'name', 'highway', 'surface', 'active', 'scenario_id', 'modified_at']
    list_filter = ['active', 'scenario_id', 'highway', 'surface']
    search_fields = ['osm_id', 'name']
    readonly_fields = ['osm_id', 'modified_at']


@admin.register(WaterOSM)
class WaterOSMAdmin(gis_admin.OSMGeoAdmin):
    list_display = ['osm_id', 'name', 'natural', 'water', 'active', 'scenario_id', 'modified_at']
    list_filter = ['active', 'scenario_id', 'natural', 'water']
    search_fields = ['osm_id', 'name']
    readonly_fields = ['osm_id', 'modified_at']


@admin.register(GreenOSM)
class GreenOSMAdmin(gis_admin.OSMGeoAdmin):
    list_display = ['osm_id', 'name', 'leisure', 'landuse', 'natural', 'active', 'scenario_id', 'modified_at']
    list_filter = ['active', 'scenario_id', 'leisure', 'landuse', 'natural']
    search_fields = ['osm_id', 'name']
    readonly_fields = ['osm_id', 'modified_at']
