# Django imports
from admin_datta.forms import RegistrationForm, LoginForm, UserPasswordChangeForm, UserPasswordResetForm, UserSetPasswordForm
from django.contrib.auth.views import LoginView, PasswordChangeView, PasswordResetConfirmView, PasswordResetView
from django.views.generic import CreateView
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.db.models import Avg
from django.utils import timezone
from django.db import transaction
from django.http import JsonResponse
from django.db.models import Sum, F
from django.db.models.functions import TruncDay
from django.views.decorators.csrf import csrf_exempt
from .models import *
from django.shortcuts import render, redirect

# Common imports
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import requests
import time
import logging

# Machine Learning imports
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.impute import SimpleImputer

# Webscraping imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def index(request):

  context = {
    'segment'  : 'index',
    #'products' : Product.objects.all()
  }
  return render(request, "pages/index.html", context)

def tables(request):
  context = {
    'segment': 'tables'
  }
  return render(request, "pages/dynamic-tables.html", context)
# views.py

def information_page(request):
 context = {
    'segment': 'information'
  }
 return render(request, "pages/information.html", context)


def faq_page(request):
 context = {
    'segment': 'faq'
  }
 return render(request, "pages/faq.html", context)

def analysis_page(request):
 context = {
    'segment': 'analysis'
  }
 return render(request, "pages/analysis.html", context)

def login_page(request):
 context = {
    'segment': 'login'
  }
 return render(request, "pages/login.html", context)

def register_page(request):
 context = {
    'segment': 'register'
  }
 return render(request, "pages/register.html", context)

def reset_page(request):
 context = {
    'segment': 'reset'
  }
 return render(request, "pages/reset_password.html", context)

def admin_dashboard_page(request):
 context = {
    'segment': 'admin_dashboard'
  }
 return render(request, "pages/admin/admin_dashboard.html", context)

def users_page(request):
 context = {
    'segment': 'users'
  }
 return render(request, "pages/admin/manage_users.html", context)

def profile_page(request):
 context = {
    'segment': 'profile'
  }
 return render(request, "pages/profile.html", context)

def setting_page(request):
 context = {
    'segment': 'setting'
  }
 return render(request, "pages/setting.html", context)

def api_data(request):
    data = list(Area.objects.values('location', 'latitude', 'longitude'))
    return JsonResponse(data, safe=False)
 
def fetch_prediction_data(request):
    try:
        today = datetime.today().date()
        prediction_dates = [today + timedelta(days=i) for i in range(7)]

        data = []
        areas = Area.objects.all()

        for area in areas:
            # Use try-except to handle the case where no predictions exist for an area
            try:
                prediction = Prediction.objects.filter(p_area_id=area).latest('prediction_id')
            except Prediction.DoesNotExist:
                # If no prediction exists, continue to the next area
                continue

            area_data = {
                'area_id': area.area_id,
                'location': area.location,
                'latitude': area.latitude,
                'longitude': area.longitude,
                'forecast_dates': [date.strftime('%Y-%m-%d') for date in prediction_dates],
                'rainfall': [getattr(prediction, f"rainfall_forecast{i+1}", 0) for i in range(7)],
                'groundwater': [getattr(prediction, f"groundwater_forecast{i+1}", 0) for i in range(7)],
                'min_temp': [getattr(prediction, f"min_temp_forecast{i+1}", 0) for i in range(7)],
                'max_temp': [getattr(prediction, f"max_temp_forecast{i+1}", 0) for i in range(7)],
                'flood_forecast': [getattr(prediction, f"flood_forecast{i+1}", 0) for i in range(7)]
            }
            data.append(area_data)

        return JsonResponse(data, safe=False)

    except Exception as e:
        # Log the exception for debugging purposes
        # You can also consider logging more details or sending these details to a monitoring system
        print(f"Error fetching prediction data: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)
        
def fetch_rainfall_data(request):
    start_date = request.GET.get('start', '2021-01-01')
    end_date = request.GET.get('end', datetime.now().strftime('%Y-%m-%d'))
    location = request.GET.get('location', 'all')

    try:
        if location == 'all':
            data = Rainfall.objects.filter(timestamp__range=[start_date, end_date]) \
                                   .annotate(date=TruncDay('timestamp')) \
                                   .values('date') \
                                   .annotate(total_rainfall=Sum('rainfall_value')) \
                                   .order_by('date')
        else:
            data = Rainfall.objects.filter(timestamp__range=[start_date, end_date], 
                                           rf_area_id=location) \
                                   .annotate(date=TruncDay('timestamp')) \
                                   .values('date', 'rainfall_value') \
                                   .order_by('timestamp')

        return JsonResponse(list(data), safe=False)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def fetch_temperature_data(request):
    start_date = request.GET.get('start', '2021-01-01')
    end_date = request.GET.get('end', datetime.now().strftime('%Y-%m-%d'))
    location = request.GET.get('location', 'all')

    try:
        if location == 'all':
            data = Temperature.objects.filter(tp_timestamp__range=[start_date, end_date]) \
                                      .annotate(date=TruncDay('tp_timestamp')) \
                                      .values('date', 'tp_min', 'tp_max') \
                                      .order_by('tp_timestamp')
        else:
            data = Temperature.objects.filter(tp_timestamp__range=[start_date, end_date], 
                                              tp_area_id=location) \
                                      .annotate(date=TruncDay('tp_timestamp')) \
                                      .values('date', 'tp_min', 'tp_max') \
                                      .order_by('tp_timestamp')

        return JsonResponse(list(data), safe=False)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    

# Dummy data for admin and user
dummy_admin = {'username': 'admin', 'password': 'admin123'}
dummy_user = {'username': 'user', 'password': 'user123'}

def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        # Check if the input credentials match the dummy admin credentials
        if username == dummy_admin['username'] and password == dummy_admin['password']:
            # Redirect to admin dashboard
            return redirect('/admin_dashboard')

        # Check if the input credentials match the dummy user credentials
        elif username == dummy_user['username'] and password == dummy_user['password']:
            # Redirect to home page (index)
            return redirect('/index')

        else:
            # Invalid user credentials, render login page with error message
            return render(request, 'pages/login.html', {'error': 'Invalid username or password'})

    # GET request, render login page
    return render(request, 'pages/login.html')


def ml():
    try:
        # Create an empty predictions dataframe with headers only
        headers = (
            [f'rainfall_forecast{i+1}' for i in range(7)] +
            [f'groundwater_forecast{i+1}' for i in range(7)] +
            [f'min_temp_forecast{i+1}' for i in range(7)] +
            [f'max_temp_forecast{i+1}' for i in range(7)] +
            [f'flood_forecast{i+1}' for i in range(7)] +
            ['area_id']
        )
        pd.DataFrame(columns=headers).to_csv('predictions.csv', index=False)
        
        # Fetch unique area IDs
        area_ids = Area.objects.values_list('area_id', flat=True).distinct()
        
        for area_id in area_ids:
            def preprocess_data(file_path, date_col, target_col, floor_value=None, cap_value=None):
                df = pd.read_csv(file_path, header=0)
                df['ds'] = pd.to_datetime(df[date_col], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
                df.rename(columns={target_col: 'y'}, inplace=True)
                df.drop(columns=[col for col in df.columns if col not in ['ds', 'y']], inplace=True)
                if floor_value is None:
                    floor_value = df['y'].min() * 1.5
                if cap_value is None:
                    cap_value = df['y'].max() * 1.5
                df['floor'] = floor_value
                df['cap'] = cap_value
                return df

            def run_prophet_model(df, clip_negative=False):
                m = Prophet(growth='logistic')
                m.fit(df)
                future = m.make_future_dataframe(periods=7)
                future['floor'] = df['floor'].iloc[0]
                future['cap'] = df['cap'].iloc[0]
                forecast = m.predict(future)
                if clip_negative:
                    forecast['yhat'] = forecast['yhat'].clip(lower=0)
                    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
                    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
                return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

            rainfall_data = Rainfall.objects.filter(rf_area_id=area_id).values('timestamp', 'rainfall_value')
            groundwater_data = Groundwater.objects.filter(gw_depot_no__dp_area_id=area_id).values('gw_timestamp').annotate(avg_groundwater=Avg('groundwater_value'))
            min_temp_data = Temperature.objects.filter(tp_area_id=area_id).values('tp_timestamp', 'tp_min')
            max_temp_data = Temperature.objects.filter(tp_area_id=area_id).values('tp_timestamp', 'tp_max')

            df_rainfall = pd.DataFrame(list(rainfall_data))
            df_groundwater = pd.DataFrame(list(groundwater_data))
            df_min_temp = pd.DataFrame(list(min_temp_data))
            df_max_temp = pd.DataFrame(list(max_temp_data))

            df_rainfall.rename(columns={'timestamp': 'Date', 'rainfall_value': 'Rainfall'}, inplace=True)
            df_groundwater.rename(columns={'gw_timestamp': 'Date', 'avg_groundwater': 'Groundwater'}, inplace=True)
            df_min_temp.rename(columns={'tp_timestamp': 'Date', 'tp_min': 'Min_Temp'}, inplace=True)
            df_max_temp.rename(columns={'tp_timestamp': 'Date', 'tp_max': 'Max_Temp'}, inplace=True)

            combined_data = pd.merge(df_rainfall, df_groundwater, on='Date', how='outer')
            combined_data = pd.merge(combined_data, df_min_temp, on='Date', how='outer')
            combined_data = pd.merge(combined_data, df_max_temp, on='Date', how='outer')
            combined_data['Date'] = pd.to_datetime(combined_data['Date'], format='%Y-%m-%d').dt.strftime('%d/%m/%Y')
            combined_data.to_csv('combined_data.csv', index=False)

            file_path = 'combined_data.csv'
            date_col = 'Date'

            df_rainfall = preprocess_data(file_path, date_col, 'Rainfall', 0, None)
            df_rainfall['floor'] = 0
            df_rainfall['cap'] = df_rainfall['y'].max()
            forecast_rainfall = run_prophet_model(df_rainfall, clip_negative=True)

            df_groundwater = preprocess_data(file_path, date_col, 'Groundwater', None, 1)
            df_groundwater['floor'] = df_groundwater['y'].min()
            df_groundwater['cap'] = 0
            forecast_groundwater = run_prophet_model(df_groundwater, clip_negative=False)

            df_min_temp = preprocess_data(file_path, date_col, 'Min_Temp', 0, None)
            df_min_temp['floor'] = df_min_temp['y'].min()
            forecast_min_temp = run_prophet_model(df_min_temp, clip_negative=False)

            df_max_temp = preprocess_data(file_path, date_col, 'Max_Temp', 0, None)
            df_max_temp['cap'] = df_max_temp['y'].min()
            forecast_max_temp = run_prophet_model(df_max_temp, clip_negative=False)

            def process_and_merge_forecasts(file_path, forecast_rainfall, forecast_groundwater, forecast_min_temp, forecast_max_temp):
                forecast_rainfall = forecast_rainfall.tail(7)
                forecast_groundwater = forecast_groundwater.tail(7)
                forecast_min_temp = forecast_min_temp.tail(7)
                forecast_max_temp = forecast_max_temp.tail(7)

                combined_forecast = pd.merge(
                    forecast_rainfall, 
                    forecast_groundwater, 
                    on='ds', 
                    how='outer',
                    suffixes=('_rainfall', '_groundwater')
                )
                combined_forecast = pd.merge(
                    combined_forecast, 
                    forecast_min_temp,
                    on='ds',
                    how='outer',
                    suffixes=('', '_min_temp')
                )
                combined_forecast = pd.merge(
                    combined_forecast, 
                    forecast_max_temp,
                    on='ds',
                    how='outer',
                    suffixes=('', '_max_temp')
                )

                combined_forecast.rename(
                    columns={
                        'ds': 'Date',
                        'yhat_rainfall': 'Rainfall', 
                        'yhat_groundwater': 'Groundwater', 
                        'yhat': 'Min_Temp', 
                        'yhat_max_temp': 'Max_Temp'
                    }, 
                    inplace=True
                )
                combined_forecast['Date'] = pd.to_datetime(combined_forecast['Date']).dt.strftime('%d/%m/%Y')
                combined_forecast = combined_forecast[['Date', 'Rainfall', 'Groundwater', 'Min_Temp', 'Max_Temp']]
                temp_df = pd.read_csv(file_path, header=0)
                temp_df.rename(columns={
                    'Time': 'Date', 
                    'rainfall': 'Rainfall', 
                    'Groundwater level': 'Groundwater',
                    'Min_Temp': 'Min_Temp', 
                    'Max_Temp': 'Max_Temp'
                }, inplace=True)
                final_df = pd.concat([temp_df, combined_forecast], ignore_index=True)
                return final_df

            df = process_and_merge_forecasts(file_path, forecast_rainfall, forecast_groundwater, forecast_min_temp, forecast_max_temp)
            standardise = ['Rainfall', 'Groundwater', 'Min_Temp', 'Max_Temp']
            scaler = StandardScaler()
            df[standardise] = scaler.fit_transform(df[standardise])
            imputer = SimpleImputer(strategy='mean')
            df[['Rainfall', 'Groundwater', 'Min_Temp', 'Max_Temp']] = imputer.fit_transform(df[['Rainfall', 'Groundwater', 'Min_Temp', 'Max_Temp']])

            def find_optimal_dbscan_params(df, feature_columns, eps_range, min_samples_range):
                X = df[feature_columns].values
                X_scaled = StandardScaler().fit_transform(X)
                best_score = -1
                best_eps = None
                best_min_samples = None
                for eps in eps_range:
                    for min_samples in min_samples_range:
                        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
                        labels = db.labels_
                        if len(set(labels)) > 1 and -1 not in labels:
                            score = silhouette_score(X_scaled, labels)
                            if score > best_score:
                                best_score = score
                                best_eps = eps
                                best_min_samples = min_samples
                return best_eps, best_min_samples

            feature_columns = ['Rainfall', 'Groundwater', 'Max_Temp']
            eps_range = np.arange(0.25, 0.75, 0.01)
            min_samples_range = range(1, 3)
            best_eps, best_min_samples = find_optimal_dbscan_params(df, feature_columns, eps_range, min_samples_range)
            X = df[['Rainfall', 'Groundwater', 'Max_Temp']].values
            db = DBSCAN(eps=best_eps, min_samples=best_min_samples).fit(X)
            labels = db.labels_
            df['Cluster_Labels'] = labels
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
            df.to_csv('df.csv', index=False)
            flood_dates = df.loc[df['Cluster_Labels'] != 0, ['Date', 'Rainfall', 'Groundwater', 'Max_Temp']]
            flood_dates = flood_dates.sort_values(by='Date')
            forecast_rainfall = forecast_rainfall.tail(7)
            forecast_groundwater = forecast_groundwater.tail(7)
            forecast_min_temp = forecast_min_temp.tail(7)
            forecast_max_temp = forecast_max_temp.tail(7)
            flood_dates_df = df.loc[df['Cluster_Labels'] != 0, ['Date']].drop_duplicates()
            flood_dates_df['Date'] = pd.to_datetime(flood_dates_df['Date'])
            rainfall_forecasts = [forecast_rainfall.iloc[i]['yhat'] for i in range(7)]
            groundwater_forecasts = [forecast_groundwater.iloc[i]['yhat'] for i in range(7)]
            min_temp_forecasts = [forecast_min_temp.iloc[i]['yhat'] for i in range(7)]
            max_temp_forecasts = [forecast_max_temp.iloc[i]['yhat'] for i in range(7)]
            flood_forecasts = [1 if forecast_rainfall.iloc[i]['ds'] in flood_dates_df['Date'].values else 0 for i in range(7)]
            adjusted_min_temps = [min(min_temp, max_temp) for min_temp, max_temp in zip(min_temp_forecasts, max_temp_forecasts)]
            data = [
                *rainfall_forecasts,
                *groundwater_forecasts,
                *adjusted_min_temps,
                *max_temp_forecasts,
                *flood_forecasts,
                area_id
            ]
            columns = (
                [f'rainfall_forecast{i+1}' for i in range(7)] +
                [f'groundwater_forecast{i+1}' for i in range(7)] +
                [f'min_temp_forecast{i+1}' for i in range(7)] +
                [f'max_temp_forecast{i+1}' for i in range(7)] +
                [f'flood_forecast{i+1}' for i in range(7)] +
                ['area_id']
            )
            data_dict = {col: [val] for col, val in zip(columns, data)}
            predictions = pd.DataFrame(data_dict)
            predictions.to_csv('predictions.csv', index=False, mode='a', header=False)

        predictions = pd.read_csv('predictions.csv', header=0)
        Prediction.objects.all().delete()
        for _, row in predictions.iterrows():
            prediction = Prediction(
                p_area_id=Area.objects.get(area_id=row['area_id']),
                groundwater_forecast1=row['groundwater_forecast1'],
                groundwater_forecast2=row['groundwater_forecast2'],
                groundwater_forecast3=row['groundwater_forecast3'],
                groundwater_forecast4=row['groundwater_forecast4'],
                groundwater_forecast5=row['groundwater_forecast5'],
                groundwater_forecast6=row['groundwater_forecast6'],
                groundwater_forecast7=row['groundwater_forecast7'],
                rainfall_forecast1=row['rainfall_forecast1'],
                rainfall_forecast2=row['rainfall_forecast2'],
                rainfall_forecast3=row['rainfall_forecast3'],
                rainfall_forecast4=row['rainfall_forecast4'],
                rainfall_forecast5=row['rainfall_forecast5'],
                rainfall_forecast6=row['rainfall_forecast6'],
                rainfall_forecast7=row['rainfall_forecast7'],
                min_temp_forecast1=row['min_temp_forecast1'],
                min_temp_forecast2=row['min_temp_forecast2'],
                min_temp_forecast3=row['min_temp_forecast3'],
                min_temp_forecast4=row['min_temp_forecast4'],
                min_temp_forecast5=row['min_temp_forecast5'],
                min_temp_forecast6=row['min_temp_forecast6'],
                min_temp_forecast7=row['min_temp_forecast7'],
                max_temp_forecast1=row['max_temp_forecast1'],
                max_temp_forecast2=row['max_temp_forecast2'],
                max_temp_forecast3=row['max_temp_forecast3'],
                max_temp_forecast4=row['max_temp_forecast4'],
                max_temp_forecast5=row['max_temp_forecast5'],
                max_temp_forecast6=row['max_temp_forecast6'],
                max_temp_forecast7=row['max_temp_forecast7'],
                flood_forecast1=row['flood_forecast1'],
                flood_forecast2=row['flood_forecast2'],
                flood_forecast3=row['flood_forecast3'],
                flood_forecast4=row['flood_forecast4'],
                flood_forecast5=row['flood_forecast5'],
                flood_forecast6=row['flood_forecast6'],
                flood_forecast7=row['flood_forecast7']
            )
            prediction.save()
        return {'status': 'success', 'predictions': predictions.to_html(classes='table table-striped', index=False)}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def webscrape():
    try:
        def webscrape_rainfall():
            latest_rainfall = Rainfall.objects.order_by('-timestamp').first()
            latest_date = latest_rainfall.timestamp
            website = 'http://111.93.109.166/CMWSSB-web/onlineMonitoringSystem/waterLevel'
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(website)
            driver.refresh()
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, 'category')))
            video = driver.find_element(By.TAG_NAME, "video")
            driver.execute_script("arguments[0].autoplay = false;", video)
            category_dropdown = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, 'category')))
            category_dropdown = Select(category_dropdown)
            category_dropdown.select_by_visible_text('Rainfall')
            location_dropdown = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, 'areaLocationId')))
            location_dropdown = Select(location_dropdown)
            current_date_time = datetime.now().date()
            current_date_time = current_date_time - timedelta(days=1)

            if latest_date < current_date_time:
                start_date = latest_date + timedelta(days=1)
                end_date = current_date_time
                area2data = []
                area_id = []
                date = []
                months = {'1': 'Jan', '2': 'Feb', '3': 'Mar', '4': 'Apr', '5': 'May', '6': 'Jun', '7': 'Jul', '8': 'Aug', '9': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}
                for location in location_dropdown.options:
                    option_value = location.get_attribute("value")
                    if option_value in ['0', '1-1', '2-1', '3-1', '4-1', '5-1', '6-1']:
                        continue
                    location_dropdown.select_by_value(option_value)
                    current_date = start_date
                    while current_date <= end_date:
                        year = current_date.year
                        month = current_date.month
                        month = months.get(str(month))
                        day = current_date.day
                        date_picker = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.XPATH, "//section[@class='content']//label[@class='input-group-text']")))
                        date_picker.click()
                        yearRange_element = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.XPATH, "//div[@class='datetimepicker-days']//th[@class='switch']"))
                        )
                        yearRange_element.click()
                        yearspecific_element = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.XPATH, "//div[@class='datetimepicker-months']//th[@class='switch']"))
                        )
                        yearspecific_element.click()
                        year_element = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable(
                                (By.XPATH, f"//div[@class='datetimepicker-years']//span[text()='{year}']"))
                        )
                        year_element.click()
                        month_element = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable(
                                (By.XPATH, f"//div[@class='datetimepicker-months']//span[text()='{month}']"))
                        )
                        month_element.click()
                        day_element = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable(
                                (By.XPATH, f"//div[@class='datetimepicker-days']//td[@class='day' and text()='{day}']"))
                        )
                        day_element.click()
                        rainfall_button = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.ID, 'checkWaterLevel')))
                        rainfall_button.click()
                        match = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.ID, 'inputLevelDiv2')))
                        try:
                            area2data.append(float(match.text))
                            area_id.append(int(option_value))
                            date.append(current_date)
                        except ValueError:
                            current_date += timedelta(days=1)
                            continue
                        current_date += timedelta(days=1)
                df_iteration_rainfall = pd.DataFrame({"rainfall_value": area2data})
                df_iteration_date = pd.DataFrame({"timestamp": date})
                df_iteration_areaid = pd.DataFrame({"rf_area_id": area_id})
                combined_data = pd.concat([df_iteration_date['timestamp'], df_iteration_rainfall['rainfall_value'], df_iteration_areaid['rf_area_id']], axis=1)
                rainfall_instances = []
                for index, row in combined_data.iterrows():
                    area_instance = Area.objects.get(area_id=row['rf_area_id'])
                    new_rainfall = Rainfall(
                        timestamp=row['timestamp'],
                        rainfall_value=row['rainfall_value'],
                        rf_area_id=area_instance
                    )
                    rainfall_instances.append(new_rainfall)
                if rainfall_instances:
                    with transaction.atomic():
                        Rainfall.objects.bulk_create(rainfall_instances)
            driver.quit()

        def webscrape_groundwater():
            latest_groundwater = Groundwater.objects.order_by('-gw_timestamp').first()
            latest_date = latest_groundwater.gw_timestamp
            website = 'http://111.93.109.166/CMWSSB-web/onlineMonitoringSystem/waterLevel'
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(website)
            driver.refresh()
            WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, 'category')))
            video = driver.find_element(By.TAG_NAME, "video")
            driver.execute_script("arguments[0].autoplay = false;", video)
            category_dropdown = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, 'category')))
            category_dropdown = Select(category_dropdown)
            category_dropdown.select_by_visible_text('Ground Water level')
            area_dropdown = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, 'areaId')))
            area_dropdown = Select(area_dropdown)
            current_date_time = datetime.now().date()
            current_date_time = current_date_time - timedelta(days=1)

            if latest_date < current_date_time:
                start_date = latest_date + timedelta(days=1)
                end_date = current_date_time
                area2data = []
                depot_id = []
                date = []
                months = {'1': 'Jan', '2': 'Feb', '3': 'Mar', '4': 'Apr', '5': 'May', '6': 'Jun', '7': 'Jul', '8': 'Aug', '9': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}
                for area in area_dropdown.options:
                    area_option_value = area.get_attribute("value")
                    if area_option_value == '0':
                        continue
                    area_dropdown.select_by_value(area_option_value)
                    location_dropdown = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.ID, 'deptId')))
                    location_dropdown = Select(location_dropdown)
                    for location in location_dropdown.options:
                        option_value = location.get_attribute("value")
                        location_dropdown.select_by_value(option_value)
                        current_date = start_date
                        while current_date <= end_date:
                            year = current_date.year
                            month = current_date.month
                            month = months.get(str(month))
                            day = current_date.day
                            date_picker = WebDriverWait(driver, 10).until(
                                EC.element_to_be_clickable((By.XPATH, "//section[@class='content']//label[@class='input-group-text']")))
                            date_picker.click()
                            yearRange_element = WebDriverWait(driver, 10).until(
                                EC.element_to_be_clickable((By.XPATH, "//div[@class='datetimepicker-days']//th[@class='switch']"))
                            )
                            yearRange_element.click()
                            yearspecific_element = WebDriverWait(driver, 10).until(
                                EC.element_to_be_clickable((By.XPATH, "//div[@class='datetimepicker-months']//th[@class='switch']"))
                            )
                            yearspecific_element.click()
                            year_element = WebDriverWait(driver, 10).until(
                                EC.element_to_be_clickable(
                                    (By.XPATH, f"//div[@class='datetimepicker-years']//span[text()='{year}']"))
                            )
                            year_element.click()
                            month_element = WebDriverWait(driver, 10).until(
                                EC.element_to_be_clickable(
                                    (By.XPATH, f"//div[@class='datetimepicker-months']//span[text()='{month}']"))
                            )
                            month_element.click()
                            day_element = WebDriverWait(driver, 10).until(
                                EC.element_to_be_clickable(
                                    (By.XPATH, f"//div[@class='datetimepicker-days']//td[@class='day' and text()='{day}']"))
                            )
                            day_element.click()
                            rainfall_button = WebDriverWait(driver, 10).until(
                                EC.element_to_be_clickable((By.ID, 'checkWaterLevel')))
                            rainfall_button.click()
                            match = WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((By.ID, 'inputLevelDiv1')))
                            try:
                                area2data.append(float(match.text))
                                date.append(current_date)
                                depot_id.append(option_value)
                            except ValueError:
                                current_date += timedelta(days=1)
                                continue
                            current_date += timedelta(days=1)
                df_iteration_groundwater = pd.DataFrame({"groundwater_value": area2data})
                df_iteration_date = pd.DataFrame({"date": date})
                df_iteration_id = pd.DataFrame({"depot_id": depot_id})
                combined_data = pd.concat([df_iteration_date['date'], df_iteration_groundwater['groundwater_value'], df_iteration_id['depot_id']], axis=1)
                groundwater_instances = []
                for index, row in combined_data.iterrows():
                    depot_instance = Depot.objects.get(depot_no=row['depot_id'])
                    new_groundwater = Groundwater(
                        gw_timestamp=row['date'],
                        groundwater_value=row['groundwater_value'],
                        gw_depot_no=depot_instance
                    )
                    groundwater_instances.append(new_groundwater)
                if groundwater_instances:
                    with transaction.atomic():
                        Groundwater.objects.bulk_create(groundwater_instances)
            driver.quit()

        def webscrape_temperature():
            latest_temperature = Temperature.objects.order_by('-tp_timestamp').first()
            latest_date = latest_temperature.tp_timestamp
            coordinates_data = {
                'areaID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                'Latitude': [13.15202524, 13.17262101, 13.15119447, 13.12339742, 13.11416046,
                             13.11126115, 13.08639463, 13.08532326, 13.03316477, 13.04169033,
                             13.0419716, 13.00571456, 13.02899838, 12.98040224, 12.90125466],
                'Longitude': [80.3028678, 80.25677016, 80.23866637, 80.26009935, 80.2885777,
                              80.24848458, 80.18617769, 80.20240682, 80.25840872, 80.23838672,
                              80.16476616, 80.20253394, 80.23071804, 80.25291808, 80.22387146]
            }
            co_df = pd.DataFrame(coordinates_data)
            current_date_time = datetime.now().date()
            current_date_time = current_date_time - timedelta(days=1)

            if latest_date < current_date_time:
                start_date = latest_date + timedelta(days=1)
                end_date = current_date_time
                mini_temp = []
                max_temp = []
                area_id = []
                date = []
                for id in range(0, 15):
                    latitude = co_df.loc[id, 'Latitude']
                    longitude = co_df.loc[id, 'Longitude']
                    current_date = start_date
                    while current_date <= end_date:
                        area_id.append(co_df['areaID'][id])
                        date.append(current_date)
                        historical_url = f'https://api.weatherstack.com/historical?access_key=8d95e3320888902c919001974be5f6b6&query={latitude},{longitude}&historical_date={current_date}&hourly=0'
                        response = requests.get(historical_url)
                        if response.status_code == 200:
                            data = response.json()
                            mini_temperature = data['historical'][f'{current_date}']['mintemp']
                            mini_temp.append(mini_temperature)
                            max_temperature = data['historical'][f'{current_date}']['maxtemp']
                            max_temp.append(max_temperature)
                        current_date += timedelta(days=1)
                        time.sleep(0.5)
                df_iteration_mini = pd.DataFrame({'min_temperature': mini_temp})
                df_iteration_max = pd.DataFrame({'max_temperature': max_temp})
                df_iteration_area = pd.DataFrame({'area_id': area_id})
                df_iteration_date = pd.DataFrame({'date': date})
                df_combine = pd.concat([df_iteration_area['area_id'], df_iteration_date['date'], df_iteration_mini['min_temperature'], df_iteration_max['max_temperature']], axis=1)
                temperature_instances = []
                for index, row in df_combine.iterrows():
                    area_instance = Area.objects.get(area_id=row['area_id'])
                    new_temperature = Temperature(
                        tp_timestamp=row['date'],
                        tp_min=row['min_temperature'],
                        tp_max=row['max_temperature'],
                        tp_area_id=area_instance
                    )
                    temperature_instances.append(new_temperature)
                if temperature_instances:
                    with transaction.atomic():
                        Temperature.objects.bulk_create(temperature_instances)
        try:
            webscrape_rainfall()
        except:
            pass
        try:
            webscrape_groundwater()
        except:
            pass
        try:
            webscrape_temperature()
        except:
            pass
        return {'status': 'success'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

@csrf_exempt
def update_data(request):
    if request.method == 'POST':
        webscrape_result = webscrape()
        ml_result = ml()
        if webscrape_result['status'] == 'success' and ml_result['status'] == 'success':
            return JsonResponse({'success': True, 'predictions': ml_result['predictions']})
        else:
            return JsonResponse({'success': False, 'error': webscrape_result.get('error', '') + ' ' + ml_result.get('error', '')})
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


