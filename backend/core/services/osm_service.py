"""
OSM (OpenStreetMap) data download and processing service.

This service handles downloading building data from OpenStreetMap
for multiple cities and processing it into the database.
"""
import requests
import json
import logging
from typing import List, Dict, Optional
from django.contrib.gis.geos import MultiPolygon, Polygon, GEOSGeometry
from django.db import transaction
from core.models import City, Building

logger = logging.getLogger(__name__)

# City coordinates (bounding boxes: [min_lat, min_lon, max_lat, max_lon])
CITY_BOUNDS = {
    'Bengaluru': {
        'bounds': [12.8, 77.4, 13.1, 77.8],
        'latitude': 12.9716,
        'longitude': 77.5946,
    },
    'Delhi': {
        'bounds': [28.4, 77.0, 28.8, 77.4],
        'latitude': 28.6139,
        'longitude': 77.2090,
    },
    'Mumbai': {
        'bounds': [18.9, 72.7, 19.3, 73.0],
        'latitude': 19.0760,
        'longitude': 72.8777,
    },
    'Chennai': {
        'bounds': [12.8, 80.1, 13.2, 80.4],
        'latitude': 13.0827,
        'longitude': 80.2707,
    },
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = 300  # seconds


def generate_overpass_query(bounds: List[float]) -> str:
    """
    Generate Overpass API query for buildings within bounding box.
    
    Args:
        bounds: [min_lat, min_lon, max_lat, max_lon]
    
    Returns:
        Overpass query string
    """
    min_lat, min_lon, max_lat, max_lon = bounds
    query = f"""
[out:json][timeout:{OVERPASS_TIMEOUT}];
(
  way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
  relation["building"]({min_lat},{min_lon},{max_lat},{max_lon});
);
out geom;
"""
    return query


def download_osm_buildings(city_name: str, bounds: Optional[List[float]] = None) -> Dict:
    """
    Download building data from OpenStreetMap for a city.
    
    Args:
        city_name: Name of the city
        bounds: Optional bounding box [min_lat, min_lon, max_lat, max_lon]
                If not provided, uses default bounds for the city
    
    Returns:
        Dictionary containing OSM response data
    
    Raises:
        requests.RequestException: If the API request fails
    """
    if bounds is None:
        if city_name not in CITY_BOUNDS:
            raise ValueError(f"Unknown city: {city_name}. Available cities: {list(CITY_BOUNDS.keys())}")
        bounds = CITY_BOUNDS[city_name]['bounds']
    
    logger.info(f"Downloading OSM building data for {city_name}...")
    query = generate_overpass_query(bounds)
    
    try:
        response = requests.post(OVERPASS_URL, data={'data': query}, timeout=OVERPASS_TIMEOUT + 60)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"Downloaded {len(data.get('elements', []))} elements for {city_name}")
        return data
    
    except requests.RequestException as e:
        logger.error(f"Error downloading OSM data for {city_name}: {str(e)}")
        raise


def process_osm_element(element: Dict) -> Optional[Dict]:
    """
    Process a single OSM element and extract building information.
    
    Args:
        element: OSM element dictionary
    
    Returns:
        Dictionary with processed building data or None if invalid
    """
    if element.get('type') not in ['way', 'relation']:
        return None
    
    # Extract OSM ID
    osm_id = element.get('id')
    if not osm_id:
        return None
    
    # Extract building tags
    tags = element.get('tags', {})
    building_type = tags.get('building', '')
    if not building_type:
        return None  # Skip if not a building
    
    # Extract height (convert to float if present)
    height = None
    height_str = tags.get('height') or tags.get('building:levels')
    if height_str:
        try:
            # Handle formats like "10", "10 m", "10m", "3 levels"
            height_str = height_str.replace('m', '').replace('levels', '').strip()
            height = float(height_str)
            # If it's levels, estimate ~3m per level
            if 'building:levels' in tags:
                height = height * 3.0
        except (ValueError, AttributeError):
            pass
    
    # Extract geometry
    geometry = None
    if 'geometry' in element:
        # Process way geometry
        coords = []
        for node in element['geometry']:
            if 'lat' in node and 'lon' in node:
                coords.append([node['lon'], node['lat']])
        
        if len(coords) >= 3:  # Need at least 3 points for a polygon
            # Close the polygon if not already closed
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            
            try:
                # Create Polygon from coordinates
                polygon = Polygon(coords)
                geometry = MultiPolygon(polygon)
            except Exception as e:
                logger.warning(f"Error creating geometry for OSM ID {osm_id}: {str(e)}")
                return None
    
    # Extract additional metadata
    metadata = {
        'name': tags.get('name', ''),
        'addr:street': tags.get('addr:street', ''),
        'addr:housenumber': tags.get('addr:housenumber', ''),
        'addr:city': tags.get('addr:city', ''),
        'building:material': tags.get('building:material', ''),
        'roof:material': tags.get('roof:material', ''),
        'roof:colour': tags.get('roof:colour', ''),
    }
    
    return {
        'osm_id': osm_id,
        'building_type': building_type,
        'height': height,
        'geometry': geometry,
        'metadata': metadata,
    }


@transaction.atomic
def process_and_store_buildings(city: City, osm_data: Dict, batch_size: int = 1000) -> Dict[str, int]:
    """
    Process OSM data and store buildings in the database.
    
    Args:
        city: City model instance
        osm_data: OSM API response data
        batch_size: Number of buildings to process in each batch
    
    Returns:
        Dictionary with statistics: {'processed': int, 'stored': int, 'skipped': int, 'errors': int}
    """
    elements = osm_data.get('elements', [])
    stats = {'processed': 0, 'stored': 0, 'skipped': 0, 'errors': 0}
    
    logger.info(f"Processing {len(elements)} OSM elements for {city.name}...")
    
    buildings_to_create = []
    
    for element in elements:
        stats['processed'] += 1
        
        try:
            processed = process_osm_element(element)
            if not processed:
                stats['skipped'] += 1
                continue
            
            # Check if building already exists
            if Building.objects.filter(osm_id=processed['osm_id']).exists():
                stats['skipped'] += 1
                continue
            
            if not processed['geometry']:
                stats['skipped'] += 1
                continue
            
            # Create Building instance
            building = Building(
                osm_id=processed['osm_id'],
                city=city,
                geometry=processed['geometry'],
                building_type=processed['building_type'],
                height=processed['height'],
                metadata=processed['metadata'],
            )
            buildings_to_create.append(building)
            
            # Batch insert
            if len(buildings_to_create) >= batch_size:
                Building.objects.bulk_create(buildings_to_create, ignore_conflicts=True)
                stats['stored'] += len(buildings_to_create)
                buildings_to_create = []
                logger.info(f"Stored {stats['stored']} buildings so far...")
        
        except Exception as e:
            stats['errors'] += 1
            logger.error(f"Error processing element {element.get('id', 'unknown')}: {str(e)}")
            continue
    
    # Insert remaining buildings
    if buildings_to_create:
        Building.objects.bulk_create(buildings_to_create, ignore_conflicts=True)
        stats['stored'] += len(buildings_to_create)
    
    logger.info(f"Completed processing for {city.name}: {stats}")
    return stats


def download_and_process_city(city_name: str, bounds: Optional[List[float]] = None) -> Dict[str, any]:
    """
    Download and process OSM building data for a city.
    
    Args:
        city_name: Name of the city
        bounds: Optional bounding box [min_lat, min_lon, max_lat, max_lon]
    
    Returns:
        Dictionary with processing results and statistics
    """
    # Get or create city
    if city_name not in CITY_BOUNDS:
        raise ValueError(f"Unknown city: {city_name}")
    
    city_info = CITY_BOUNDS[city_name]
    city, created = City.objects.get_or_create(
        name=city_name,
        defaults={
            'latitude': city_info['latitude'],
            'longitude': city_info['longitude'],
            'country': 'India',
        }
    )
    
    if created:
        logger.info(f"Created new city: {city_name}")
    else:
        logger.info(f"Using existing city: {city_name}")
    
    # Download OSM data
    osm_data = download_osm_buildings(city_name, bounds)
    
    # Process and store buildings
    stats = process_and_store_buildings(city, osm_data)
    
    return {
        'city': city_name,
        'city_id': city.id,
        'osm_elements': len(osm_data.get('elements', [])),
        'processing_stats': stats,
    }


def download_and_process_all_cities() -> List[Dict[str, any]]:
    """
    Download and process OSM building data for all supported cities.
    
    Returns:
        List of dictionaries with processing results for each city
    """
    results = []
    
    for city_name in CITY_BOUNDS.keys():
        try:
            result = download_and_process_city(city_name)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {city_name}: {str(e)}")
            results.append({
                'city': city_name,
                'error': str(e),
            })
    
    return results


class OSMService:
    """Object-oriented wrapper for OSM service functions."""
    
    def __init__(self):
        self.overpass_url = OVERPASS_URL
        self.timeout = OVERPASS_TIMEOUT
    
    def download_buildings(self, min_lon: float, min_lat: float, max_lon: float, max_lat: float, city: City) -> List[Building]:
        """
        Download and store buildings for a city within the given bounding box.
        
        Args:
            min_lon: Minimum longitude
            min_lat: Minimum latitude
            max_lon: Maximum longitude
            max_lat: Maximum latitude
            city: City model instance
        
        Returns:
            List of created Building instances
        """
        bounds = [min_lat, min_lon, max_lat, max_lon]
        query = generate_overpass_query(bounds)
        
        try:
            logger.info(f"Downloading buildings for {city.name} (bbox: {bounds})")
            response = requests.post(self.overpass_url, data={'data': query}, timeout=self.timeout + 60)
            response.raise_for_status()
            
            osm_data = response.json()
            stats = process_and_store_buildings(city, osm_data)
            
            logger.info(f"Downloaded buildings for {city.name}: {stats}")
            
            # Return the buildings created for this city
            return list(Building.objects.filter(city=city).order_by('-id')[:stats['stored']])
        
        except requests.RequestException as e:
            logger.error(f"Error downloading buildings for {city.name}: {str(e)}")
            raise


