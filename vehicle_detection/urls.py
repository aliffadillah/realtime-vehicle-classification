# vehicle_detection/urls.py
from django.contrib import admin
from django.urls import path
from detection import views  # This import assumes views.py is in the detection directory

urlpatterns = [
    path('admin/', admin.site.urls),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('', views.index, name='index'),
]
