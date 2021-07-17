from Models.classes import model
from django.db import models

class Number(models.Model):
    value = models.SmallIntegerField()


class Execution(models.Model):
    experiment_name = models.CharField(max_length=100, db_index=True)
    execution_file = models.BinaryField()
    configurations = models.JSONField()
    results = models.JSONField()
