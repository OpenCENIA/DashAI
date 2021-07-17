# Generated by Django 3.2.4 on 2021-06-29 22:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Experimenter', '0002_exp_model'),
    ]

    operations = [
        migrations.CreateModel(
            name='Execution',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('experiment_name', models.CharField(db_index=True, max_length=100)),
                ('execution_file', models.BinaryField()),
                ('configurations', models.JSONField()),
                ('results', models.JSONField()),
            ],
        ),
        migrations.DeleteModel(
            name='Exp_Model',
        ),
    ]
