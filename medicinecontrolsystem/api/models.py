from django.db import models
from django.contrib.auth.models import UserManager, AbstractBaseUser


class Sensor(AbstractBaseUser):
    name = models.CharField('Name', max_length=255, blank=True, default='')
    model = models.CharField('Model', max_length=255, blank=True, default='')
    manufacturer = models.CharField('Manufacturer', max_length=255, blank=True, default='')
    serial = models.BigIntegerField('Serial number', unique=True)
    password = models.CharField('Password', max_length=255)

    USERNAME_FIELD = 'serial'
    REQUIRED_FIELDS = ['password']

    objects = UserManager()


class SensorData(models.Model):
    temperature = models.FloatField('Temperature')
    humidity = models.FloatField('Humidity')
    datetime = models.DateTimeField('Datetime', auto_now_add=True)
    sensor = models.ForeignKey(
        Sensor, related_name='data', on_delete=models.CASCADE)
