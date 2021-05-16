from rest_framework.viewsets import ModelViewSet

from .models import Sensor
from .serializers import SensorSerializer


class SensorDataView(ModelViewSet):

    queryset = Sensor.objects.all()
    serializer_class = SensorSerializer
