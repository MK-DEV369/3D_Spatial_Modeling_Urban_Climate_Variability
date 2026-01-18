from django.core.management.base import BaseCommand
from core.models import City


class Command(BaseCommand):
    help = 'Populate initial cities data for Urban Climate Platform'

    def handle(self, *args, **kwargs):
        cities_data = [
            {
                'name': 'Bengaluru',
                'country': 'India',
                'latitude': 12.9716,
                'longitude': 77.5946,
                'metadata': {
                    'population': 12425000,
                    'area_km2': 741,
                    'planning_type': 'unplanned',
                    'timezone': 'Asia/Kolkata'
                }
            },
            {
                'name': 'Dubai',
                'country': 'UAE',
                'latitude': 25.2048,
                'longitude': 55.2708,
                'metadata': {
                    'population': 3331000,
                    'area_km2': 4114,
                    'planning_type': 'planned',
                    'timezone': 'Asia/Dubai'
                }
            },
            {
                'name': 'Amsterdam',
                'country': 'Netherlands',
                'latitude': 52.3676,
                'longitude': 4.9041,
                'metadata': {
                    'population': 872680,
                    'area_km2': 219,
                    'planning_type': 'planned',
                    'timezone': 'Europe/Amsterdam'
                }
            },
            {
                'name': 'Mumbai',
                'country': 'India',
                'latitude': 19.0760,
                'longitude': 72.8777,
                'metadata': {
                    'population': 20411000,
                    'area_km2': 603,
                    'planning_type': 'mixed',
                    'timezone': 'Asia/Kolkata'
                }
            },
            {
                'name': 'Delhi',
                'country': 'India',
                'latitude': 28.7041,
                'longitude': 77.1025,
                'metadata': {
                    'population': 30291000,
                    'area_km2': 1484,
                    'planning_type': 'mixed',
                    'timezone': 'Asia/Kolkata'
                }
            },
            {
                'name': 'Singapore',
                'country': 'Singapore',
                'latitude': 1.3521,
                'longitude': 103.8198,
                'metadata': {
                    'population': 5686000,
                    'area_km2': 728,
                    'planning_type': 'planned',
                    'timezone': 'Asia/Singapore'
                }
            },
            {
                'name': 'Chennai',
                'country': 'India',
                'latitude': 13.0827,
                'longitude': 80.2707,
                'metadata': {
                    'population': 10971000,
                    'area_km2': 426,
                    'planning_type': 'unplanned',
                    'timezone': 'Asia/Kolkata'
                }
            },
            {
                'name': 'Hyderabad',
                'country': 'India',
                'latitude': 17.3850,
                'longitude': 78.4867,
                'metadata': {
                    'population': 10268000,
                    'area_km2': 650,
                    'planning_type': 'mixed',
                    'timezone': 'Asia/Kolkata'
                }
            },
        ]

        created_count = 0
        updated_count = 0

        for city_data in cities_data:
            city, created = City.objects.update_or_create(
                name=city_data['name'],
                country=city_data['country'],
                defaults={
                    'latitude': city_data['latitude'],
                    'longitude': city_data['longitude'],
                    'metadata': city_data['metadata']
                }
            )
            if created:
                created_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'Created city: {city.name}, {city.country}')
                )
            else:
                updated_count += 1
                self.stdout.write(
                    self.style.WARNING(f'Updated city: {city.name}, {city.country}')
                )

        self.stdout.write(
            self.style.SUCCESS(
                f'\nCompleted: {created_count} cities created, {updated_count} cities updated'
            )
        )
