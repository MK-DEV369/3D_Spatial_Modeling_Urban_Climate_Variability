"""
Management command to sync external data (weather, pollution) for all cities.
"""
import logging
from django.core.management.base import BaseCommand
from core.services.data_integration_service import DataIntegrationService
from core.models import City

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Sync weather and pollution data for all cities from external APIs'

    def add_arguments(self, parser):
        parser.add_argument(
            '--city',
            type=str,
            help='Sync data for a specific city only'
        )

    def handle(self, *args, **options):
        city_name = options.get('city')
        
        service = DataIntegrationService()
        
        if city_name:
            # Sync single city
            try:
                city = City.objects.get(name__iexact=city_name)
                self.stdout.write(self.style.NOTICE(f'Syncing data for {city.name}...'))
                result = service.sync_city_data(city)
                
                if result['errors']:
                    for error in result['errors']:
                        self.stdout.write(self.style.WARNING(f'  ⚠ {error}'))
                
                self.stdout.write(self.style.SUCCESS(
                    f'  ✓ Climate data: {"stored" if result["climate"] else "failed"}\n'
                    f'  ✓ Pollution data: {"stored" if result["pollution"] else "failed"}'
                ))
            except City.DoesNotExist:
                self.stdout.write(self.style.ERROR(f'City "{city_name}" not found'))
                return
        else:
            # Sync all cities
            cities_count = City.objects.count()
            if cities_count == 0:
                self.stdout.write(self.style.WARNING('No cities found in database'))
                self.stdout.write('Run: python manage.py seed_cities first')
                return
            
            self.stdout.write(self.style.NOTICE(f'Syncing data for {cities_count} cities...'))
            results = service.sync_all_cities()
            
            success_count = sum(1 for r in results if r['climate'] or r['pollution'])
            
            self.stdout.write(f'\n{"-" * 60}')
            for result in results:
                status = '✓' if (result['climate'] or result['pollution']) else '✗'
                self.stdout.write(f'{status} {result["city"]}')
                if result['errors']:
                    for error in result['errors']:
                        self.stdout.write(f'    {error}')
            
            self.stdout.write(f'\n{"-" * 60}')
            self.stdout.write(self.style.SUCCESS(
                f'\nSuccessfully synced {success_count}/{cities_count} cities'
            ))
            
            if success_count < cities_count:
                self.stdout.write(self.style.WARNING(
                    '\nNote: Set OPENWEATHER_API_KEY environment variable for real data'
                ))
