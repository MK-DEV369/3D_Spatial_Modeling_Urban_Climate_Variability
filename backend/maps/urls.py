from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'buildings', views.BuildingsOSMViewSet, basename='buildings')
router.register(r'roads', views.RoadsOSMViewSet, basename='roads')
router.register(r'water', views.WaterOSMViewSet, basename='water')
router.register(r'green', views.GreenOSMViewSet, basename='green')

urlpatterns = [
    path('', include(router.urls)),
]
