from django.urls import path
from . import views

urlpatterns = [
    path('', views.NLP_view, name='NLP_view'),
]