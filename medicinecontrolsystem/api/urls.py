from django.urls import path, include

from rest_framework.routers import DefaultRouter
from rest_framework_nested.routers import NestedDefaultRouter
from rest_framework_simplejwt.views import TokenRefreshView

from .views import SensorDataView, SensorView, RegisterSensorView, ObtainTokenPairViewWithSensorId

sensor_router = DefaultRouter()
sensor_router.register(r'sensors', SensorView, basename='sensors')
sensor_router.register(r'register', RegisterSensorView, basename='register')

sensor_data_router = NestedDefaultRouter(sensor_router, r'sensors')
sensor_data_router.register(r'data', SensorDataView, basename='sensor data')

urlpatterns = [
    path('', include(sensor_router.urls)),
    path('', include(sensor_data_router.urls)),
    path(r'login/', ObtainTokenPairViewWithSensorId.as_view()),
    path(r'login/refresh', TokenRefreshView.as_view()),
]
