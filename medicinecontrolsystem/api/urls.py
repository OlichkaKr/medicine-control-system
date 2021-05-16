from django.urls import path, include

from rest_framework.routers import DefaultRouter

from .views import SensorDataView

router = DefaultRouter()
router.register(r'sensor', SensorDataView, basename='sensor')

urlpatterns = [
    path('', include(router.urls))
]
