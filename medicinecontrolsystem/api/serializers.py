from django.contrib.auth import hashers

from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

from .models import Sensor, SensorData


class SensorDataSerializer(serializers.ModelSerializer):

    class Meta:
        model = SensorData
        fields = '__all__'

    def to_representation(self, instance):
        rep = super(SensorDataSerializer, self).to_representation(instance)
        rep['sensor'] = instance.sensor.serial if rep['sensor'] else None
        return rep


class SensorSerializer(serializers.ModelSerializer):
    data = SensorDataSerializer(many=True, allow_null=True, required=False)

    class Meta:
        model = Sensor
        fields = ['id', 'name', 'model', 'manufacturer', 'serial', 'password', 'data']

    def create(self, validated_data):
        validated_data['password'] = SensorSerializer._hash_password(validated_data['password'])
        sensors_data = validated_data.pop('data') if 'data' in validated_data else []
        sensor = Sensor.objects.create(**validated_data)

        for sensor_data in sensors_data:
            SensorData.objects.create(sensor=sensor, **sensor_data)

        return sensor

    def update(self, instance, validated_data):
        validated_data['password'] = SensorSerializer._hash_password(validated_data['password'])
        for field, value in validated_data.items():
            setattr(instance, field, value)
        return instance

    @staticmethod
    def _hash_password(password: str):
        return hashers.make_password(password)


class TokenObtainPairSerializerWithSensorId(TokenObtainPairSerializer):

    def validate(self, attrs):
        data = super().validate(attrs)

        refresh = self.get_token(self.user)

        data['refresh'] = str(refresh)
        data['access'] = str(refresh.access_token)
        data['sensor_id'] = self.user.id

        return data
