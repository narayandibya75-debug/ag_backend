import os

import requests

# Set OPENWEATHER_API_KEY in your hosting environment (never commit the key).
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"


def get_weather(city):
    """
    Return (temperature_C, humidity_percent, rain_last_hour_mm) for a city.

    Raises:
        RuntimeError: the API key is not configured.
        ValueError: the city was not found or the key was rejected.
        requests.RequestException: the weather service could not be reached.
    """
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENWEATHER_API_KEY is not set on the server.")

    response = requests.get(
        WEATHER_URL,
        params={"q": city, "appid": api_key, "units": "metric"},
        timeout=10,
    )

    if response.status_code == 404:
        raise ValueError(f"City '{city}' was not found. Check the spelling.")
    if response.status_code == 401:
        raise RuntimeError("OpenWeather rejected the API key.")
    response.raise_for_status()

    data = response.json()
    temperature = float(data["main"]["temp"])
    humidity = float(data["main"]["humidity"])
    # OpenWeather includes a "rain" object only while it is raining.
    rain = float(data.get("rain", {}).get("1h", 0.0))

    return temperature, humidity, rain
