from django.contrib.gis.db import models

class BuildingsOSM(models.Model):
    osm_id = models.BigIntegerField(primary_key=True)
    name = models.TextField(null=True)
    building = models.TextField(null=True)
    geom = models.GeometryField(srid=3857)

    active = models.BooleanField(default=True)
    scenario_id = models.TextField(default='baseline')
    modified_at = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'buildings_osm'


class RoadsOSM(models.Model):
    osm_id = models.BigIntegerField(primary_key=True)
    name = models.TextField(null=True)
    highway = models.TextField(null=True)
    surface = models.TextField(null=True)
    layer = models.IntegerField(null=True)
    geom = models.GeometryField(srid=3857)

    active = models.BooleanField(default=True)
    scenario_id = models.TextField(default='baseline')
    modified_at = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'roads_osm'


class WaterOSM(models.Model):
    osm_id = models.BigIntegerField(primary_key=True)
    name = models.TextField(null=True)
    natural = models.TextField(null=True)
    water = models.TextField(null=True)
    geom = models.GeometryField(srid=3857)

    active = models.BooleanField(default=True)
    scenario_id = models.TextField(default='baseline')
    modified_at = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'water_osm'


class GreenOSM(models.Model):
    osm_id = models.BigIntegerField(primary_key=True)
    name = models.TextField(null=True)
    leisure = models.TextField(null=True)
    landuse = models.TextField(null=True)
    natural = models.TextField(null=True)
    geom = models.GeometryField(srid=3857)

    active = models.BooleanField(default=True)
    scenario_id = models.TextField(default='baseline')
    modified_at = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'green_osm'
