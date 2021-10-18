from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('init_experiment', views.init_experiment, name='experiment'),
    path('configure_experiment', views.configure_experiment, name='configure'),
    path('execute_experiment', views.execute_experiment, name='execute'),
    path('test', views.test, name='test'),
]