"""
Django management command to download and process OSM building data.

Usage:
    python manage.py download_osm_data [city_name]
    python manage.py download_osm_data --all
"""
from django.core.management.base import BaseCommand
from core.services.osm_service import download_and_process_city, download_and_process_all_cities


class Command(BaseCommand):
    help = 'Download and process OSM building data for cities'

    def add_arguments(self, parser):
        parser.add_argument(
            'city',
            nargs='?',
            type=str,
            help='City name (Bengaluru, Delhi, Mumbai, Chennai)',
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Download data for all supported cities',
        )

    def handle(self, *args, **options):
        if options['all']:
            self.stdout.write(self.style.SUCCESS('Downloading OSM data for all cities...'))
            results = download_and_process_all_cities()
            
            for result in results:
                if 'error' in result:
                    self.stdout.write(
                        self.style.ERROR(f"Failed to process {result['city']}: {result['error']}")
                    )
                else:
                    stats = result['processing_stats']
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"{result['city']}: Processed {stats['processed']} elements, "
                            f"stored {stats['stored']} buildings, "
                            f"skipped {stats['skipped']}, "
                            f"errors {stats['errors']}"
                        )
                    )
        else:
            city_name = options.get('city')
            if not city_name:
                self.stdout.write(self.style.ERROR('Please provide a city name or use --all'))
                return
            
            self.stdout.write(self.style.SUCCESS(f'Downloading OSM data for {city_name}...'))
            try:
                result = download_and_process_city(city_name)
                stats = result['processing_stats']
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Successfully processed {city_name}:\n"
                        f"  OSM elements: {result['osm_elements']}\n"
                        f"  Processed: {stats['processed']}\n"
                        f"  Stored: {stats['stored']}\n"
                        f"  Skipped: {stats['skipped']}\n"
                        f"  Errors: {stats['errors']}"
                    )
                )
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error processing {city_name}: {str(e)}')
                )

