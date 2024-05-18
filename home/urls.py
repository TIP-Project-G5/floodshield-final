from django.urls import path
from django.contrib.auth import views as auth_views

from . import views

urlpatterns = [
    path('', views.index,  name='index'),
    path('index/', views.index,  name='index'),
    path('tables/', views.tables, name='tables'),
    path('information/', views.information_page, name='information'),
    path('faq/', views.faq_page, name='information'),
    path('analysis/', views.analysis_page, name='information'),
    path('login/', views.login, name='login'),  # Updated path for login
    path('register/', views.register_page, name='register'),
    path('reset/', views.reset_page, name='reset_password'),
    path('admin_dashboard/', views.admin_dashboard_page, name='admin_dashboard'),
    path('users/', views.users_page, name='users'),
    path('profile/', views.profile_page, name='profile'),
    path('setting/', views.setting_page, name='setting'),
    path('api/data/', views.api_data, name='api_data'),
    path('fetch_prediction_data/', views.fetch_prediction_data, name='fetch_prediction_data'),
    path('api/fetch_rainfall_data', views.fetch_rainfall_data, name='fetch_rainfall_data'),
    path('api/fetch_temperature_data', views.fetch_temperature_data, name='fetch_temperature_data'),
    path('ml/', views.ml, name='ml'),
    path('webscrape/', views.webscrape, name='webscrape'),
    path('update_data/', views.update_data, name='update_data'),  # Add this line
]
