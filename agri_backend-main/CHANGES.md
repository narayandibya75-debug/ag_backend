# Backend fixes

- `/fertilizer` no longer returns 500. The request now accepts `ph` (optional) and `crop` (optional); pH is passed to `recommend_fertilizer`.
- `/irrigation_ai` no longer returns 500. `get_weather()` returns `(temperature, humidity, rain)`, and `main.py` now unpacks it correctly.
- Unknown city now returns 404 with a clear message, and a weather-service outage returns 502. Before, errors silently produced fake 25 °C / 50% / 0 mm weather and wrong irrigation advice.
- The OpenWeather API key was hard-coded. It is now read from the `OPENWEATHER_API_KEY` environment variable. **Revoke the old key** (6d28...c12c5) because it was shared in a zip.
- `/predict_disease` now passes the confidence to `get_cure`, so `cure.confidence` is correct. `severity` is still a simple confidence-based label; do not treat it as real disease severity.
- `procfile.txt` pointed to a non-existent `api_server` module. Replaced with `Procfile` running `main:app`.

## Deploying
Set these environment variables on Render/your host:
- `OPENWEATHER_API_KEY` = your new key
