"""
DRF ViewSets for CRUD operations on OSM layers (buildings, roads, water, green spaces).
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.contrib.gis.geos import Polygon
from django.contrib.gis.db.models.functions import Transform
from django.utils import timezone
from django.db import connection

from .models import BuildingsOSM, RoadsOSM, WaterOSM, GreenOSM
from .serializers import (
    BuildingsOSMSerializer,
    RoadsOSMSerializer,
    WaterOSMSerializer,
    GreenOSMSerializer,
)


def parse_bbox(bbox_str):
    """
    Parse bbox string format: minLon,minLat,maxLon,maxLat (EPSG:4326)
    """
    try:
        min_lon, min_lat, max_lon, max_lat = map(float, bbox_str.split(','))
        bbox = Polygon.from_bbox((min_lon, min_lat, max_lon, max_lat))
        bbox.srid = 4326
        return bbox
    except (ValueError, TypeError):
        return None


class BuildingsOSMViewSet(viewsets.ModelViewSet):
    """
    ViewSet for CRUD operations on BuildingsOSM.
    
    Supports:
    - List/Create/Retrieve/Update/Delete buildings
    - Bbox-based filtering for map rendering
    - Scenario-based filtering
    """
    queryset = BuildingsOSM.objects.all()
    serializer_class = BuildingsOSMSerializer
    lookup_field = 'osm_id'
    
    def get_queryset(self):
        """Filter by active status and scenario."""
        queryset = BuildingsOSM.objects.all()
        active = self.request.query_params.get('active', None)
        scenario = self.request.query_params.get('scenario', None)
        
        if active is not None:
            queryset = queryset.filter(active=active.lower() == 'true')
        
        if scenario:
            queryset = queryset.filter(scenario_id=scenario)
        
        return queryset
    
    def perform_update(self, serializer):
        """Update modified_at timestamp on update."""
        serializer.save(modified_at=timezone.now())
    
    def perform_create(self, serializer):
        """Set modified_at on create."""
        serializer.save(modified_at=timezone.now())
    
    @action(detail=False, methods=['get'])
    def bbox(self, request):
        """
        Get buildings within bounding box.
        
        Query params:
        - bbox: minLon,minLat,maxLon,maxLat (required)
        - scenario: scenario_id (default: 'baseline')
        - active: true/false (default: true)
        """
        bbox_param = request.query_params.get('bbox')
        scenario = request.query_params.get('scenario', 'baseline')
        active = request.query_params.get('active', 'true')
        
        if not bbox_param:
            return Response(
                {'error': 'bbox parameter required (minLon,minLat,maxLon,maxLat)'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        bbox_geom = parse_bbox(bbox_param)
        if bbox_geom is None:
            return Response(
                {'error': 'Invalid bbox format. Expected: minLon,minLat,maxLon,maxLat'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        queryset = self.get_queryset().filter(active=active.lower() == 'true', scenario_id=scenario)
        queryset = queryset.filter(geom__intersects=Transform(bbox_geom, 3857))
        
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'type': 'FeatureCollection',
            'features': serializer.data
        })


class RoadsOSMViewSet(viewsets.ModelViewSet):
    """ViewSet for CRUD operations on RoadsOSM."""
    queryset = RoadsOSM.objects.all()
    serializer_class = RoadsOSMSerializer
    lookup_field = 'osm_id'
    
    def get_queryset(self):
        """Filter by active status and scenario."""
        queryset = RoadsOSM.objects.all()
        active = self.request.query_params.get('active', None)
        scenario = self.request.query_params.get('scenario', None)
        
        if active is not None:
            queryset = queryset.filter(active=active.lower() == 'true')
        
        if scenario:
            queryset = queryset.filter(scenario_id=scenario)
        
        return queryset
    
    def perform_update(self, serializer):
        """Update modified_at timestamp on update."""
        serializer.save(modified_at=timezone.now())
    
    def perform_create(self, serializer):
        """Set modified_at on create."""
        serializer.save(modified_at=timezone.now())
    
    @action(detail=False, methods=['get'])
    def bbox(self, request):
        """Get roads within bounding box."""
        bbox_param = request.query_params.get('bbox')
        scenario = request.query_params.get('scenario', 'baseline')
        active = request.query_params.get('active', 'true')
        
        if not bbox_param:
            return Response(
                {'error': 'bbox parameter required (minLon,minLat,maxLon,maxLat)'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        bbox_geom = parse_bbox(bbox_param)
        if bbox_geom is None:
            return Response(
                {'error': 'Invalid bbox format. Expected: minLon,minLat,maxLon,maxLat'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        queryset = self.get_queryset().filter(active=active.lower() == 'true', scenario_id=scenario)
        queryset = queryset.filter(geom__intersects=Transform(bbox_geom, 3857))
        
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'type': 'FeatureCollection',
            'features': serializer.data
        })


class WaterOSMViewSet(viewsets.ModelViewSet):
    """ViewSet for CRUD operations on WaterOSM."""
    queryset = WaterOSM.objects.all()
    serializer_class = WaterOSMSerializer
    lookup_field = 'osm_id'
    
    def get_queryset(self):
        """Filter by active status and scenario."""
        queryset = WaterOSM.objects.all()
        active = self.request.query_params.get('active', None)
        scenario = self.request.query_params.get('scenario', None)
        
        if active is not None:
            queryset = queryset.filter(active=active.lower() == 'true')
        
        if scenario:
            queryset = queryset.filter(scenario_id=scenario)
        
        return queryset
    
    def perform_update(self, serializer):
        """Update modified_at timestamp on update."""
        serializer.save(modified_at=timezone.now())
    
    def perform_create(self, serializer):
        """Set modified_at on create."""
        serializer.save(modified_at=timezone.now())
    
    @action(detail=False, methods=['get'])
    def bbox(self, request):
        """Get water features within bounding box."""
        bbox_param = request.query_params.get('bbox')
        scenario = request.query_params.get('scenario', 'baseline')
        active = request.query_params.get('active', 'true')
        
        if not bbox_param:
            return Response(
                {'error': 'bbox parameter required (minLon,minLat,maxLon,maxLat)'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        bbox_geom = parse_bbox(bbox_param)
        if bbox_geom is None:
            return Response(
                {'error': 'Invalid bbox format. Expected: minLon,minLat,maxLon,maxLat'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        queryset = self.get_queryset().filter(active=active.lower() == 'true', scenario_id=scenario)
        queryset = queryset.filter(geom__intersects=Transform(bbox_geom, 3857))
        
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'type': 'FeatureCollection',
            'features': serializer.data
        })


class GreenOSMViewSet(viewsets.ModelViewSet):
    """ViewSet for CRUD operations on GreenOSM."""
    queryset = GreenOSM.objects.all()
    serializer_class = GreenOSMSerializer
    lookup_field = 'osm_id'
    
    def get_queryset(self):
        """Filter by active status and scenario."""
        queryset = GreenOSM.objects.all()
        active = self.request.query_params.get('active', None)
        scenario = self.request.query_params.get('scenario', None)
        
        if active is not None:
            queryset = queryset.filter(active=active.lower() == 'true')
        
        if scenario:
            queryset = queryset.filter(scenario_id=scenario)
        
        return queryset
    
    def perform_update(self, serializer):
        """Update modified_at timestamp on update."""
        serializer.save(modified_at=timezone.now())
    
    def perform_create(self, serializer):
        """Set modified_at on create."""
        serializer.save(modified_at=timezone.now())
    
    @action(detail=False, methods=['get'])
    def bbox(self, request):
        """Get green spaces within bounding box."""
        bbox_param = request.query_params.get('bbox')
        scenario = request.query_params.get('scenario', 'baseline')
        active = request.query_params.get('active', 'true')
        
        if not bbox_param:
            return Response(
                {'error': 'bbox parameter required (minLon,minLat,maxLon,maxLat)'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        bbox_geom = parse_bbox(bbox_param)
        if bbox_geom is None:
            return Response(
                {'error': 'Invalid bbox format. Expected: minLon,minLat,maxLon,maxLat'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        queryset = self.get_queryset().filter(active=active.lower() == 'true', scenario_id=scenario)
        queryset = queryset.filter(geom__intersects=Transform(bbox_geom, 3857))
        
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'type': 'FeatureCollection',
            'features': serializer.data
        })
