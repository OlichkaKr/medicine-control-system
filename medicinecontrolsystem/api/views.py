from rest_framework import status
from rest_framework.mixins import CreateModelMixin
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet, GenericViewSet
from rest_framework.permissions import AllowAny, IsAuthenticated, IsAuthenticatedOrReadOnly
from rest_framework_simplejwt.views import TokenObtainPairView

from .models import Sensor, SensorData
from .serializers import SensorSerializer, SensorDataSerializer, TokenObtainPairSerializerWithSensorId


class SensorDataView(ModelViewSet):

    permission_classes = [IsAuthenticated]
    queryset = SensorData.objects.all()
    serializer_class = SensorDataSerializer

    def list(self, request, *args, **kwargs):
        if kwargs.get('nested_1_pk') and kwargs.get('nested_1_pk') == str(request.user.id):
            queryset = self.filter_queryset(self.get_queryset().filter(sensor_id=str(request.user.id)))

            page = self.paginate_queryset(queryset)
            if page is not None:
                serializer = self.get_serializer(page, many=True)
                return self.get_paginated_response(serializer.data)

            serializer = self.get_serializer(queryset, many=True)
            return Response(serializer.data)
        else:
            return Response({"error": "Sensor can not get data of another sensor."})

    def retrieve(self, request, *args, **kwargs):
        if kwargs.get('nested_1_pk') and kwargs.get('nested_1_pk') == str(request.user.id):
            instance = self.get_object()
            if instance.sensor_id and instance.sensor_id == request.user.id:
                serializer = self.get_serializer(instance)
                return Response(serializer.data)

        return Response({"error": "Sensor can not get data of another sensor."})

    def create(self, request, *args, **kwargs):
        if kwargs.get('nested_1_pk') and kwargs.get('nested_1_pk') == str(request.user.id):
            if isinstance(request.data, dict):
                request.data.update({'sensor': str(request.user.id)})
            else:
                request.data._mutable = True
                request.data.update({'sensor': str(request.user.id)})
                request.data._mutable = False
            return super().create(request, args, kwargs)

        return Response({"error": "Sensor can not change data of another sensor."})

    def update(self, request, *args, **kwargs):
        if kwargs.get('nested_1_pk') and kwargs.get('nested_1_pk') == str(request.user.id):
            instance = self.get_object()
            if instance.sensor_id and instance.sensor_id == request.user.id:
                if isinstance(request.data, dict):
                    request.data.update({'sensor': str(request.user.id)})
                else:
                    request.data._mutable = True
                    request.data.update({'sensor': str(request.user.id)})
                    request.data._mutable = False
                return super().update(request, args, kwargs)

        return Response({"error": "Sensor can not change data of another sensor."})

    def destroy(self, request, *args, **kwargs):
        if kwargs.get('nested_1_pk') and kwargs.get('nested_1_pk') == str(request.user.id):
            instance = self.get_object()
            if instance.sensor_id and instance.sensor_id == request.user.id:
                instance.delete()
                return Response(status=status.HTTP_204_NO_CONTENT)

        return Response({"error": "Sensor can not delete data of another sensor."})


class SensorView(ModelViewSet):

    permission_classes = (IsAuthenticatedOrReadOnly,)
    queryset = Sensor.objects.all()
    serializer_class = SensorSerializer

    def retrieve(self, request, *args, **kwargs):
        if not request.user.id:
            return Response({"detail": "Authentication credentials were not provided."})
        if kwargs.get('pk') and kwargs.get('pk') == str(request.user.id):
            return super().retrieve(request, args, kwargs)
        else:
            return Response({"error": "Sensor can not get data of another sensor."})

    def update(self, request, *args, **kwargs):
        if kwargs.get('pk') and kwargs.get('pk') == str(request.user.id):
            return super().update(request, args, kwargs)
        else:
            return Response({"error": "Sensor can not change data of another sensor."})

    def destroy(self, request, *args, **kwargs):
        if kwargs.get('pk') and kwargs.get('pk') == str(request.user.id):
            return super().destroy(request, args, kwargs)
        else:
            return Response({"error": "Sensor can not delete data of another sensor."})


class RegisterSensorView(CreateModelMixin, GenericViewSet):

    permission_classes = (AllowAny,)
    queryset = Sensor.objects.all()
    serializer_class = SensorSerializer


class ObtainTokenPairViewWithSensorId(TokenObtainPairView):
    permission_classes = (AllowAny,)
    serializer_class = TokenObtainPairSerializerWithSensorId
