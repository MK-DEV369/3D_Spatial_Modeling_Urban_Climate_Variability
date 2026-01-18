from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from core.models import City, ClimateData, TrafficData, PollutionData, Scenario, Prediction
# Note: Building model removed - use maps.models.BuildingsOSM instead
from core.services.climate_service import (
    generate_weather_forecast,
    generate_climate_projection,
    save_weather_forecast_to_db,
    save_climate_projection_to_db,
)
from core.services.traffic_service import (
    generate_traffic_prediction,
    save_traffic_prediction_to_db,
)
from core.services.pollution_service import (
    generate_pollution_prediction,
    save_pollution_prediction_to_db,
)
from core.services.scenario_service import (
    run_scenario_simulation,
    get_scenario_summary,
)
from core.tasks import (
    generate_weather_forecast_task,
    generate_climate_projection_task,
    generate_traffic_prediction_task,
    generate_pollution_prediction_task,
)
from .serializers import (
    CitySerializer, ClimateDataSerializer,
    TrafficDataSerializer, PollutionDataSerializer, ScenarioSerializer,
    PredictionSerializer
)


@api_view(['GET'])
def api_root(request, format=None):
    """
    API root endpoint listing all available endpoints.
    """
    return Response({
        'cities': 'http://localhost:8000/api/cities/',
        'scenarios': 'http://localhost:8000/api/scenarios/',
        'predictions': 'http://localhost:8000/api/predictions/',
        'buildings': 'http://localhost:8000/api/buildings/',
        'roads': 'http://localhost:8000/api/roads/',
        'water': 'http://localhost:8000/api/water/',
        'green': 'http://localhost:8000/api/green/',
    })


class CityViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for viewing cities.
    """
    queryset = City.objects.all()
    serializer_class = CitySerializer

    # Buildings endpoint removed - use /api/buildings/ instead from maps app
    # The Building model in core.models is deprecated in favor of maps.models.BuildingsOSM

    @action(detail=True, methods=['get'])
    def climate(self, request, pk=None):
        """
        Get historical climate data for a city.
        """
        city = self.get_object()
        climate_data = ClimateData.objects.filter(city=city).order_by('-timestamp')
        serializer = ClimateDataSerializer(climate_data, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def traffic(self, request, pk=None):
        """
        Get historical traffic data for a city.
        """
        city = self.get_object()
        traffic_data = TrafficData.objects.filter(city=city).order_by('-timestamp')
        serializer = TrafficDataSerializer(traffic_data, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def pollution(self, request, pk=None):
        """
        Get historical pollution data for a city.
        """
        city = self.get_object()
        pollution_data = PollutionData.objects.filter(city=city).order_by('-timestamp')
        serializer = PollutionDataSerializer(pollution_data, many=True)
        return Response(serializer.data)


class ScenarioViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing scenarios.
    """
    queryset = Scenario.objects.all()
    serializer_class = ScenarioSerializer

    @action(detail=True, methods=['get'])
    def predictions(self, request, pk=None):
        """
        Get all predictions for a scenario.
        """
        scenario = self.get_object()
        predictions = Prediction.objects.filter(scenario=scenario).order_by('-timestamp')
        serializer = PredictionSerializer(predictions, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def run(self, request, pk=None):
        """
        Run complete scenario simulation using all ML models.
        
        Expected payload:
        {
            "async": bool (default: True),
            "include_models": list (optional, e.g., ["climate", "traffic", "pollution"])
        }
        """
        scenario = self.get_object()
        async_mode = request.data.get('async', True)
        include_models = request.data.get('include_models', None)
        
        if async_mode:
            # For async mode, we'll use individual tasks
            # This is a simplified version - in production, you might want
            # a single orchestration task that runs all models
            from core.services.scenario_service import get_projection_years, get_prediction_horizon_hours
            
            projection_years = get_projection_years(scenario.time_horizon)
            prediction_hours = get_prediction_horizon_hours(scenario.time_horizon)
            
            task_ids = []
            
            # Run climate prediction
            if include_models is None or 'climate' in include_models:
                if projection_years > 0:
                    task = generate_climate_projection_task.delay(
                        city_id=scenario.city.id,
                        scenario_id=scenario.id,
                        projection_years=projection_years,
                    )
                    task_ids.append({'model': 'climate', 'task_id': task.id})
                else:
                    task = generate_weather_forecast_task.delay(
                        city_id=scenario.city.id,
                        forecast_days=prediction_hours // 24,
                        scenario_id=scenario.id,
                    )
                    task_ids.append({'model': 'weather', 'task_id': task.id})
            
            # Run traffic prediction (only for short-term)
            if projection_years == 0 and (include_models is None or 'traffic' in include_models):
                task = generate_traffic_prediction_task.delay(
                    city_id=scenario.city.id,
                    prediction_hours=prediction_hours,
                    scenario_id=scenario.id,
                )
                task_ids.append({'model': 'traffic', 'task_id': task.id})
            
            # Run pollution prediction (only for short-term)
            if projection_years == 0 and (include_models is None or 'pollution' in include_models):
                task = generate_pollution_prediction_task.delay(
                    city_id=scenario.city.id,
                    prediction_hours=prediction_hours,
                    scenario_id=scenario.id,
                )
                task_ids.append({'model': 'pollution', 'task_id': task.id})
            
            return Response({
                'message': 'Scenario simulation started',
                'scenario_id': scenario.id,
                'tasks': task_ids,
                'status': 'processing',
            }, status=status.HTTP_202_ACCEPTED)
        else:
            # Run synchronously
            results = run_scenario_simulation(
                scenario,
                include_models=include_models,
                async_mode=False,
            )
            return Response({
                'message': 'Scenario simulation completed',
                **results,
            }, status=status.HTTP_200_OK)
    
    @action(detail=True, methods=['get'])
    def summary(self, request, pk=None):
        """
        Get summary of all predictions for a scenario.
        """
        scenario = self.get_object()
        summary = get_scenario_summary(scenario)
        return Response(summary, status=status.HTTP_200_OK)


class PredictionViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for viewing predictions.
    """
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
    
    @action(detail=False, methods=['post'])
    def weather(self, request):
        """
        Generate weather forecast using GraphCast (1-15 days).
        
        Expected payload:
        {
            "city_id": int,
            "forecast_days": int (1-15),
            "scenario_id": int (optional),
            "async": bool (default: True)
        }
        """
        city_id = request.data.get('city_id')
        forecast_days = request.data.get('forecast_days', 7)
        scenario_id = request.data.get('scenario_id')
        async_mode = request.data.get('async', True)
        
        if not city_id:
            return Response(
                {'error': 'city_id is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if forecast_days < 1 or forecast_days > 15:
            return Response(
                {'error': 'forecast_days must be between 1 and 15'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        city = get_object_or_404(City, pk=city_id)
        scenario = None
        if scenario_id:
            scenario = get_object_or_404(Scenario, pk=scenario_id)
        
        if async_mode:
            # Run asynchronously
            task = generate_weather_forecast_task.delay(
                city_id=city.id,
                forecast_days=forecast_days,
                scenario_id=scenario.id if scenario else None,
            )
            return Response({
                'message': 'Weather forecast generation started',
                'task_id': task.id,
                'status': 'processing',
            }, status=status.HTTP_202_ACCEPTED)
        else:
            # Run synchronously
            forecast = generate_weather_forecast(city, forecast_days=forecast_days)
            predictions = save_weather_forecast_to_db(city, forecast, scenario=scenario)
            return Response({
                'message': 'Weather forecast generated',
                'forecast': forecast,
                'predictions_created': len(predictions),
            }, status=status.HTTP_200_OK)
    
    @action(detail=False, methods=['post'])
    def climate(self, request):
        """
        Generate climate projection using ClimaX (years).
        
        Expected payload:
        {
            "city_id": int,
            "scenario_id": int,
            "projection_years": int (1-50, default: 10),
            "async": bool (default: True)
        }
        """
        city_id = request.data.get('city_id')
        scenario_id = request.data.get('scenario_id')
        projection_years = request.data.get('projection_years', 10)
        async_mode = request.data.get('async', True)
        
        if not city_id or not scenario_id:
            return Response(
                {'error': 'city_id and scenario_id are required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if projection_years < 1 or projection_years > 50:
            return Response(
                {'error': 'projection_years must be between 1 and 50'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        city = get_object_or_404(City, pk=city_id)
        scenario = get_object_or_404(Scenario, pk=scenario_id)
        
        if async_mode:
            # Run asynchronously
            task = generate_climate_projection_task.delay(
                city_id=city.id,
                scenario_id=scenario.id,
                projection_years=projection_years,
            )
            return Response({
                'message': 'Climate projection generation started',
                'task_id': task.id,
                'status': 'processing',
            }, status=status.HTTP_202_ACCEPTED)
        else:
            # Run synchronously
            projection = generate_climate_projection(
                city,
                scenario=scenario,
                projection_years=projection_years,
            )
            predictions = save_climate_projection_to_db(city, projection, scenario)
            return Response({
                'message': 'Climate projection generated',
                'projection': projection,
                'predictions_created': len(predictions),
            }, status=status.HTTP_200_OK)
    
    @action(detail=False, methods=['post'])
    def traffic(self, request):
        """
        Generate traffic prediction.
        
        Expected payload:
        {
            "city_id": int,
            "prediction_hours": int (1-168, default: 24),
            "scenario_id": int (optional),
            "async": bool (default: True)
        }
        """
        city_id = request.data.get('city_id')
        prediction_hours = request.data.get('prediction_hours', 24)
        scenario_id = request.data.get('scenario_id')
        async_mode = request.data.get('async', True)
        
        if not city_id:
            return Response(
                {'error': 'city_id is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if prediction_hours < 1 or prediction_hours > 168:
            return Response(
                {'error': 'prediction_hours must be between 1 and 168 (1 week)'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        city = get_object_or_404(City, pk=city_id)
        scenario = None
        if scenario_id:
            scenario = get_object_or_404(Scenario, pk=scenario_id)
        
        if async_mode:
            # Run asynchronously
            task = generate_traffic_prediction_task.delay(
                city_id=city.id,
                prediction_hours=prediction_hours,
                scenario_id=scenario.id if scenario else None,
            )
            return Response({
                'message': 'Traffic prediction generation started',
                'task_id': task.id,
                'status': 'processing',
            }, status=status.HTTP_202_ACCEPTED)
        else:
            # Run synchronously
            prediction = generate_traffic_prediction(
                city,
                prediction_hours=prediction_hours,
                scenario=scenario,
            )
            predictions = save_traffic_prediction_to_db(city, prediction, scenario=scenario)
            return Response({
                'message': 'Traffic prediction generated',
                'prediction': prediction,
                'predictions_created': len(predictions),
            }, status=status.HTTP_200_OK)
    
    @action(detail=False, methods=['post'])
    def pollution(self, request):
        """
        Generate pollution prediction.
        
        Expected payload:
        {
            "city_id": int,
            "prediction_hours": int (1-168, default: 24),
            "scenario_id": int (optional),
            "async": bool (default: True)
        }
        """
        city_id = request.data.get('city_id')
        prediction_hours = request.data.get('prediction_hours', 24)
        scenario_id = request.data.get('scenario_id')
        async_mode = request.data.get('async', True)
        
        if not city_id:
            return Response(
                {'error': 'city_id is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if prediction_hours < 1 or prediction_hours > 168:
            return Response(
                {'error': 'prediction_hours must be between 1 and 168 (1 week)'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        city = get_object_or_404(City, pk=city_id)
        scenario = None
        if scenario_id:
            scenario = get_object_or_404(Scenario, pk=scenario_id)
        
        if async_mode:
            # Run asynchronously
            task = generate_pollution_prediction_task.delay(
                city_id=city.id,
                prediction_hours=prediction_hours,
                scenario_id=scenario.id if scenario else None,
            )
            return Response({
                'message': 'Pollution prediction generation started',
                'task_id': task.id,
                'status': 'processing',
            }, status=status.HTTP_202_ACCEPTED)
        else:
            # Run synchronously
            prediction = generate_pollution_prediction(
                city,
                prediction_hours=prediction_hours,
                scenario=scenario,
            )
            predictions = save_pollution_prediction_to_db(city, prediction, scenario=scenario)
            return Response({
                'message': 'Pollution prediction generated',
                'prediction': prediction,
                'predictions_created': len(predictions),
            }, status=status.HTTP_200_OK)
