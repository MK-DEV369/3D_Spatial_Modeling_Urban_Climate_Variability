from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'cities', views.CityViewSet, basename='city')
router.register(r'scenarios', views.ScenarioViewSet, basename='scenario')
router.register(r'predictions', views.PredictionViewSet, basename='prediction')

urlpatterns = [
    path('', include(router.urls)),
]

