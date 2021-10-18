from django.db.models.deletion import CASCADE
from Models.classes import model
from django.db import models

class Experiment(models.Model):
    task_type = models.CharField(max_length=30)
    task_parameters = models.JSONField()


class Execution(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    execution_model = models.CharField(max_length=30)
    execution_file = models.BinaryField()
    parameters = models.JSONField()
    train_results = models.JSONField()
    test_results = models.JSONField()
