from django.db import models


class Sensor(models.Model):

    temperature = models.FloatField('Temperature')
    humidity = models.FloatField('Humidity')
