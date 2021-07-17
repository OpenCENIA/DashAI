from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('make_experiment', views.make_experiment, name='make_experiment'),
    path('run_experiment', views.run_experiment, name='run_experiment'),
    path('test', views.test, name='test'),
]