from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import os
import requests
import json

app = Flask(__name__)
CORS(app)

current_dir = os.path.dirname(os.path.abspath(__file__))
templates_path = os.path.join(current_dir, 'templates')

if not os.path.exists(templates_path):
    os.makedirs(templates_path)

OPENWEATHER_API_KEY = "5fa7a3ebad625d4c6718dfded81b53bd"
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

class AdvancedWeatherModel:
    def __init__(self):
        self.is_trained = True
        self.location_cache = {}
        
    def get_current_weather(self, city, country="IN"):
        try:
            location_key = f"{city}_{country}"
            if location_key in self.location_cache:
                return self.location_cache[location_key]
                
            query = f"{city},{country}"
                
            url = f"{OPENWEATHER_BASE_URL}/weather?q={query}&appid={OPENWEATHER_API_KEY}&units=metric"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                weather_data = {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'condition': data['weather'][0]['main'].lower(),
                    'description': data['weather'][0]['description'],
                    'wind_speed': data['wind']['speed'],
                    'city': data['name'],
                    'country': data['sys']['country'],
                    'lat': data['coord']['lat'],
                    'lon': data['coord']['lon'],
                    'source': 'api'
                }
                self.location_cache[location_key] = weather_data
                return weather_data
            else:
                return self.get_fallback_weather(city)
                
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return self.get_fallback_weather(city)
    
    def get_fallback_weather(self, city="Mumbai"):
        # Indian weather patterns
        indian_temperatures = {
            "Mumbai": 28 + np.random.uniform(-3, 5),
            "Delhi": 25 + np.random.uniform(-5, 7),
            "Bangalore": 23 + np.random.uniform(-4, 6),
            "Chennai": 30 + np.random.uniform(-3, 4),
            "Kolkata": 27 + np.random.uniform(-4, 5),
            "Hyderabad": 26 + np.random.uniform(-4, 6),
            "Pune": 24 + np.random.uniform(-5, 7),
            "Jaipur": 26 + np.random.uniform(-6, 8),
            "Lucknow": 24 + np.random.uniform(-5, 7),
            "Ahmedabad": 27 + np.random.uniform(-5, 8)
        }
        
        temp = indian_temperatures.get(city, 25 + np.random.uniform(-5, 5))
        
        return {
            'temperature': temp,
            'humidity': 65 + np.random.uniform(-20, 20),
            'pressure': 1010 + np.random.uniform(-10, 10),
            'condition': np.random.choice(['sunny', 'cloudy', 'partly cloudy']),
            'description': 'simulated data',
            'wind_speed': 8 + np.random.uniform(0, 5),
            'city': city,
            'country': 'IN',
            'source': 'fallback'
        }
    
    def predict_future_weather(self, date, current_weather):
        day_of_year = date.timetuple().tm_yday
        
        # Adjust base temperature for Indian climate
        base_temp = 20 + 8 * np.sin(2 * np.pi * day_of_year / 365)
        
        current_temp = current_weather['temperature']
        current_humidity = current_weather['humidity']
        current_pressure = current_weather['pressure']
        
        days_ahead = (date - datetime.now()).days
        temperature = current_temp + (base_temp - current_temp) * (days_ahead / 30)
        
        temperature += np.random.uniform(-3, 3)
        
        # Adjust conditions for Indian weather patterns
        if current_humidity > 80 and temperature > 35:
            condition = 'rainy'
        elif current_humidity > 70:
            condition = 'cloudy'
        elif temperature > 35:
            condition = 'sunny'
        else:
            condition = 'partly cloudy'
        
        return {
            'date': date.strftime('%Y-%m-%d'),
            'temperature': round(temperature, 1),
            'condition': condition,
            'humidity': max(30, min(90, current_humidity + np.random.uniform(-10, 10))),
            'pressure': max(1000, min(1020, current_pressure + np.random.uniform(-5, 5))),
            'wind_speed': current_weather['wind_speed'] + np.random.uniform(-2, 2)
        }
    
    def generate_forecast(self, city, country="IN", days=7):
        current_weather = self.get_current_weather(city, country)
        
        forecasts = []
        for i in range(days):
            forecast_date = datetime.now() + timedelta(days=i)
            forecast = self.predict_future_weather(forecast_date, current_weather)
            forecasts.append(forecast)
        
        return {
            'location': {
                'city': current_weather['city'],
                'country': current_weather['country'],
                'coordinates': {
                    'lat': current_weather.get('lat', 0),
                    'lon': current_weather.get('lon', 0)
                }
            },
            'current_weather': current_weather,
            'forecasts': forecasts,
            'generated_at': datetime.now().isoformat(),
            'data_source': current_weather['source']
        }

weather_model = AdvancedWeatherModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/weather/current', methods=['POST'])
def get_current_weather():
    try:
        data = request.get_json()
        city = data.get('city', 'Mumbai')
        
        weather_data = weather_model.get_current_weather(city)
        
        return jsonify({
            'success': True,
            'weather': weather_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/weather/forecast', methods=['POST'])
def get_weather_forecast():
    try:
        data = request.get_json()
        city = data.get('city', 'Mumbai')
        days = int(data.get('days', 7))
        
        forecast_data = weather_model.generate_forecast(city, days=days)
        
        return jsonify({
            'success': True,
            'forecast': forecast_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/locations/search', methods=['GET'])
def search_locations():
    query = request.args.get('q', '').lower()
    
    indian_cities = [
        {'city': 'Mumbai', 'country': 'IN', 'country_name': 'India'},
        {'city': 'Delhi', 'country': 'IN', 'country_name': 'India'},
        {'city': 'Bangalore', 'country': 'IN', 'country_name': 'India'},
        {'city': 'Chennai', 'country': 'IN', 'country_name': 'India'},
        {'city': 'Kolkata', 'country': 'IN', 'country_name': 'India'},
        {'city': 'Hyderabad', 'country': 'IN', 'country_name': 'India'},
        {'city': 'Pune', 'country': 'IN', 'country_name': 'India'},
        {'city': 'Ahmedabad', 'country': 'IN', 'country_name': 'India'},
        {'city': 'Jaipur', 'country': 'IN', 'country_name': 'India'},
        {'city': 'Lucknow', 'country': 'IN', 'country_name': 'India'}
    ]
    
    if query:
        filtered_cities = [city for city in indian_cities 
                          if query in city['city'].lower()]
    else:
        filtered_cities = indian_cities
    
    return jsonify({
        'success': True,
        'locations': filtered_cities
    })

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'Weather forecast application is running',
        'api_configured': True,
        'location_scope': 'India only'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)