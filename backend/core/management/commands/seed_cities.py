"""
Management command to seed database with Indian and planned cities.
Downloads OSM building data for each city and populates the database.
"""
import logging
from django.core.management.base import BaseCommand
from django.contrib.gis.geos import Point
from core.models import City
from core.services.osm_service import OSMService

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Seed database with cities (Bengaluru, Delhi, Mumbai, Chennai, Dubai, Amsterdam)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--cities',
            nargs='+',
            type=str,
            default=['all'],
            help='Specify cities to seed: bengaluru, delhi, mumbai, chennai, dubai, amsterdam, or all'
        )
        parser.add_argument(
            '--skip-buildings',
            action='store_true',
            help='Skip downloading building data (only create city records)'
        )

    def handle(self, *args, **options):
        cities_to_seed = options['cities']
        skip_buildings = options['skip_buildings']

        # City definitions with coordinates and metadata
        city_data = {
            'bengaluru': {
                'name': 'Bengaluru',
                'country': 'India',
                'latitude': 12.9716,
                'longitude': 77.5946,
                'population': 12639000,
                'area': 741.0,  # sq km
                'planning_type': 'unplanned',
                'bbox': (77.4601, 12.8341, 77.7845, 13.1401),  # (min_lon, min_lat, max_lon, max_lat)
            },
            'delhi': {
                'name': 'Delhi',
                'country': 'India',
                'latitude': 28.7041,
                'longitude': 77.1025,
                'population': 32941000,
                'area': 1484.0,
                'planning_type': 'mixed',
                'bbox': (76.8385, 28.4041, 77.3465, 28.8834),
            },
            'mumbai': {
                'name': 'Mumbai',
                'country': 'India',
                'latitude': 19.0760,
                'longitude': 72.8777,
                'population': 21357000,
                'area': 603.0,
                'planning_type': 'mixed',
                'bbox': (72.7746, 18.8930, 72.9810, 19.2704),
            },
            'chennai': {
                'name': 'Chennai',
                'country': 'India',
                'latitude': 13.0827,
                'longitude': 80.2707,
                'population': 11324000,
                'area': 426.0,
                'planning_type': 'mixed',
                'bbox': (80.1203, 12.8342, 80.3210, 13.2340),
            },
            'dubai': {
                'name': 'Dubai',
                'country': 'UAE',
                'latitude': 25.2048,
                'longitude': 55.2708,
                'population': 3604000,
                'area': 4114.0,
                'planning_type': 'planned',
                'bbox': (54.8935, 24.7730, 55.6670, 25.4360),
            },
            'amsterdam': {
                'name': 'Amsterdam',
                'country': 'Netherlands',
                'latitude': 52.3676,
                'longitude': 4.9041,
                'population': 1158000,
                'area': 219.0,
                'planning_type': 'planned',
                'bbox': (4.7282, 52.2785, 5.0790, 52.4310),
            },
        }

        # Determine which cities to process
        if 'all' in cities_to_seed:
            cities_to_process = list(city_data.keys())
        else:
            cities_to_process = [c.lower() for c in cities_to_seed if c.lower() in city_data]

        if not cities_to_process:
            self.stdout.write(self.style.ERROR('No valid cities specified'))
            return

        self.stdout.write(self.style.SUCCESS(f'Seeding cities: {", ".join(cities_to_process)}'))

        osm_service = OSMService()

        for city_key in cities_to_process:
            city_info = city_data[city_key]
            self.stdout.write(f'\n{"-" * 60}')
            self.stdout.write(self.style.NOTICE(f'Processing: {city_info["name"]}'))

            # Create or update city record
            city, created = City.objects.update_or_create(
                name=city_info['name'],
                defaults={
                    'country': city_info['country'],
                    'location': Point(city_info['longitude'], city_info['latitude']),
                    'population': city_info['population'],
                    'area': city_info['area'],
                }
            )

            if created:
                self.stdout.write(self.style.SUCCESS(f'✓ Created city: {city.name}'))
            else:
                self.stdout.write(self.style.WARNING(f'✓ Updated city: {city.name}'))

            # Download and process building data
            if not skip_buildings:
                self.stdout.write(f'  Downloading building data for {city.name}...')
                try:
                    bbox = city_info['bbox']
                    buildings = osm_service.download_buildings(
                        min_lon=bbox[0],
                        min_lat=bbox[1],
                        max_lon=bbox[2],
                        max_lat=bbox[3],
                        city=city
                    )
                    
                    count = len(buildings) if buildings else 0
                    self.stdout.write(self.style.SUCCESS(f'  ✓ Downloaded {count} buildings'))
                    
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f'  ✗ Error downloading buildings: {str(e)}'))
                    logger.error(f'Error processing {city.name}: {str(e)}', exc_info=True)
            else:
                self.stdout.write(self.style.NOTICE('  Skipping building download'))

        self.stdout.write(f'\n{"-" * 60}')
        self.stdout.write(self.style.SUCCESS(f'\n✓ Successfully processed {len(cities_to_process)} cities'))
