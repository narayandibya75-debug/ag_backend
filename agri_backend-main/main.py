import os
import io
import re
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import requests
import joblib
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image

import gdown

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fertilizer_logic import recommend_fertilizer
from disease_cure import get_cure
from irrigation_model import get_weather


# ============================================================
# CONFIGURATION
# ============================================================

LEAF_MODEL_ID = "1dobP1idkTpZ2JQap1aj6Yvcpnqw5S1-z"
CLASSES_ID = "1YGZu2J0dYTQCYeWKTFrxyNPx_lPsxkQM"

CROP_MODEL_ID = "1cZvhsTBkXqij2mqIhvK7MDosfOQjCba1"
YIELD_MODEL_ID = "1LOEAAL4nzUNb_X5S1B7HaQX5ZZZeyHxG"

SOIL_MODEL_ID = "1bXDiPYOXAFKcxVMIyylZa_TcYorufDtA"
IRRIGATION_MODEL_ID = "1ny4qXolifwvh2hZAjiYwsCzAZOEDpi62"

LEAF_MODEL_PATH = "leaf_model.pth"
CLASSES_PATH = "classes.pth"

CROP_MODEL_PATH = "crop_model.pkl"
YIELD_MODEL_PATH = "yield_model.pkl"

SOIL_ML_MODEL_PATH = "soil_ml_model.pkl"
IRRIGATION_ML_MODEL_PATH = "irrigation_model.pkl"


def log(message: str) -> None:
    """Print and flush immediately so Render shows progress live."""
    print(message, flush=True)


# ============================================================
# GOOGLE DRIVE MODEL DOWNLOAD
# ============================================================

def _is_valid_download(path: str) -> bool:
    """Return True only when a non-empty file exists."""
    return (
        os.path.exists(path)
        and os.path.isfile(path)
        and os.path.getsize(path) > 0
    )


def _looks_like_html(path: str) -> bool:
    """
    Google Drive may return an HTML login/error/confirmation page
    instead of the requested binary file.
    """
    if not _is_valid_download(path):
        return False

    try:
        with open(path, "rb") as file:
            sample = file.read(4096).lower()

        return (
            b"<html" in sample
            or b"<!doctype html" in sample
            or b"<head" in sample
            or (b"google drive" in sample and b"<" in sample)
        )
    except Exception:
        return False


def _remove_quietly(*paths: str) -> None:
    for path in paths:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _save_response_to_file(response: requests.Response, output: str) -> None:
    """Stream a successful binary HTTP response to disk safely."""
    temp_path = output + ".part"
    _remove_quietly(temp_path)

    content_type = response.headers.get("content-type", "").lower()

    if "text/html" in content_type:
        raise RuntimeError(
            "Google returned an HTML page instead of the model file."
        )

    total_bytes = 0

    with open(temp_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
                total_bytes += len(chunk)

    if total_bytes == 0:
        raise RuntimeError("Google returned an empty response.")

    os.replace(temp_path, output)


def _extract_confirm_token(text: str) -> Optional[str]:
    """
    Extract Google's confirmation token from an HTML response.
    Google Drive sometimes returns a confirmation page for larger files.
    """
    patterns = [
        r'name="confirm"\s+value="([^"]+)"',
        r'name="confirm"\s*value="([^"]+)"',
        r"confirm=([0-9A-Za-z_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    return None


def _google_drive_requests_download(file_id: str, output: str) -> None:
    """
    Fallback download using requests, with Google's
    confirmation-token flow.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        )
    })

    url = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}

    log("⬇️ Trying Google Drive direct download...")

    response = session.get(
        url,
        params=params,
        stream=True,
        timeout=120,
        allow_redirects=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()

    # Direct binary response
    if "text/html" not in content_type:
        _save_response_to_file(response, output)

        if _is_valid_download(output) and not _looks_like_html(output):
            return

        raise RuntimeError("Google Drive returned an invalid file.")

    # HTML confirmation page
    html = response.text
    confirm_token = _extract_confirm_token(html)

    if not confirm_token:
        match = re.search(r'href="([^"]+download[^"]+)"', html)

        if match:
            download_url = match.group(1).replace("&amp;", "&")

            response2 = session.get(
                download_url,
                stream=True,
                timeout=120,
                allow_redirects=True,
            )
            response2.raise_for_status()
            _save_response_to_file(response2, output)

            if _is_valid_download(output) and not _looks_like_html(output):
                return

        raise RuntimeError(
            "Google Drive returned an HTML page and "
            "no download confirmation token could be found."
        )

    # Confirmation-token request
    confirm_params = {
        "export": "download",
        "id": file_id,
        "confirm": confirm_token,
    }

    response2 = session.get(
        url,
        params=confirm_params,
        stream=True,
        timeout=120,
        allow_redirects=True,
    )
    response2.raise_for_status()
    _save_response_to_file(response2, output)

    if not _is_valid_download(output):
        raise RuntimeError(
            "Google Drive confirmation download produced an empty file."
        )

    if _looks_like_html(output):
        raise RuntimeError(
            "Google Drive confirmation download returned HTML "
            "instead of the model."
        )


def download_google_drive_model(file_id: str, output: str) -> None:
    """
    Download a model from Google Drive.

    Order:
        1. Existing local file
        2. gdown using file ID
        3. gdown using standard Drive URL
        4. requests + Google Drive confirmation flow
    """

    if _is_valid_download(output) and not _looks_like_html(output):
        log(
            f"✓ Using existing {output} "
            f"({os.path.getsize(output) / (1024 * 1024):.2f} MB)"
        )
        return

    log(f"\n⬇️ Downloading {output} from Google Drive...")
    _remove_quietly(output, output + ".part")

    # Attempt 1: gdown with file ID
    try:
        log(f"⬇️ Trying gdown for {output}...")
        downloaded = gdown.download(id=file_id, output=output, quiet=False)

        if (
            downloaded
            and _is_valid_download(output)
            and not _looks_like_html(output)
        ):
            log(
                f"✓ Successfully downloaded {output} "
                f"({os.path.getsize(output) / (1024 * 1024):.2f} MB)"
            )
            return

    except Exception as exc:
        log(f"⚠️ gdown ID download failed for {output}: {exc}")

    _remove_quietly(output, output + ".part")

    # Attempt 2: gdown using standard Drive URL
    try:
        drive_url = f"https://drive.google.com/file/d/{file_id}/view"
        log(f"⬇️ Trying gdown Drive URL for {output}...")
        downloaded = gdown.download(url=drive_url, output=output, quiet=False)

        if (
            downloaded
            and _is_valid_download(output)
            and not _looks_like_html(output)
        ):
            log(
                f"✓ Successfully downloaded {output} "
                f"({os.path.getsize(output) / (1024 * 1024):.2f} MB)"
            )
            return

    except Exception as exc:
        log(f"⚠️ gdown URL download failed for {output}: {exc}")

    _remove_quietly(output, output + ".part")

    # Attempt 3: requests fallback
    try:
        _google_drive_requests_download(file_id, output)

        if _is_valid_download(output) and not _looks_like_html(output):
            log(
                f"✓ Successfully downloaded {output} "
                f"({os.path.getsize(output) / (1024 * 1024):.2f} MB)"
            )
            return

    except Exception as exc:
        log(f"⚠️ Google Drive direct download failed for {output}: {exc}")

    _remove_quietly(output, output + ".part")

    raise RuntimeError(
        f"Failed to download {output} from Google Drive "
        f"(file ID: {file_id}). Verify the file exists and is shared as "
        f"'Anyone with the link' (Viewer): "
        f"https://drive.google.com/file/d/{file_id}/view"
    )


# ============================================================
# MODEL STATE (populated in the background after startup)
# ============================================================

state = {
    "ready": False,
    "stage": "starting",
    "error": None,
}

crop_model = None
yield_model = None
model = None
classes = []
soil_ml_model = None
irrigation_ml_model = None

disease_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def require_ready() -> None:
    """Raise 503 while models are still loading (or failed to load)."""
    if not state["ready"]:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Models not ready (stage: {state['stage']}, "
                f"error: {state['error']})"
            ),
        )


def load_everything() -> None:
    """
    Download and load every model. Runs in a background thread so the
    web server can bind its port immediately.
    """
    global crop_model, yield_model, model, classes
    global soil_ml_model, irrigation_ml_model

    try:
        log("\n" + "=" * 60)
        log("DOWNLOADING AGRIAI MODELS")
        log("=" * 60)

        state["stage"] = "downloading"

        required = [
            (LEAF_MODEL_ID, LEAF_MODEL_PATH),
            (CLASSES_ID, CLASSES_PATH),
            (CROP_MODEL_ID, CROP_MODEL_PATH),
            (YIELD_MODEL_ID, YIELD_MODEL_PATH),
        ]

        for file_id, path in required:
            state["stage"] = f"downloading {path}"
            download_google_drive_model(file_id, path)

        # Optional models: failures here must not stop startup
        optional = [
            (SOIL_MODEL_ID, SOIL_ML_MODEL_PATH),
            (IRRIGATION_MODEL_ID, IRRIGATION_ML_MODEL_PATH),
        ]

        for file_id, path in optional:
            try:
                state["stage"] = f"downloading {path} (optional)"
                download_google_drive_model(file_id, path)
            except Exception as exc:
                log(f"⚠️ Optional model {path} unavailable: {exc}")

        log("MODEL DOWNLOADS COMPLETE")

        # ----------------------------------------------------
        # Load models
        # ----------------------------------------------------

        state["stage"] = "loading"

        crop_model = joblib.load(CROP_MODEL_PATH)
        log("✓ crop_model.pkl loaded")

        yield_model = joblib.load(YIELD_MODEL_PATH)
        log("✓ yield_model.pkl loaded")

        if _is_valid_download(SOIL_ML_MODEL_PATH):
            try:
                soil_ml_model = joblib.load(SOIL_ML_MODEL_PATH)
                log(f"✓ Loaded {SOIL_ML_MODEL_PATH}")
            except Exception as exc:
                log(f"⚠️ Could not load {SOIL_ML_MODEL_PATH}: {exc}")

        if _is_valid_download(IRRIGATION_ML_MODEL_PATH):
            try:
                irrigation_ml_model = joblib.load(IRRIGATION_ML_MODEL_PATH)
                log(f"✓ Loaded {IRRIGATION_ML_MODEL_PATH}")
            except Exception as exc:
                log(f"⚠️ Could not load {IRRIGATION_ML_MODEL_PATH}: {exc}")

        loaded_classes = torch.load(CLASSES_PATH, map_location="cpu")

        if isinstance(loaded_classes, torch.Tensor):
            loaded_classes = loaded_classes.tolist()

        classes = list(loaded_classes)
        log(f"✓ Loaded {len(classes)} disease classes")

        leaf = models.resnet18(weights=None)
        leaf.fc = nn.Linear(leaf.fc.in_features, len(classes))
        leaf.load_state_dict(
            torch.load(LEAF_MODEL_PATH, map_location="cpu")
        )
        leaf.eval()
        model = leaf
        log("✓ leaf_model.pth loaded")

        state["stage"] = "ready"
        state["ready"] = True
        log("✓ All models ready")

    except Exception as exc:
        state["stage"] = "failed"
        state["error"] = repr(exc)
        log(f"❌ Model loading failed: {exc!r}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=load_everything, daemon=True).start()
    yield


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AgriAI Smart Farming API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SENSOR STATE
# ============================================================

_sensor_lock = threading.Lock()

_latest_sensor_data = {
    "moisture": None,
    "temperature": None,
    "ec": None,
    "ph": None,
    "N": None,
    "P": None,
    "K": None,
    "rainfall": None,
    "updated_at": None,
    "source": "ESP32 / RS485 soil sensor",
}


# ============================================================
# REQUEST MODELS
# ============================================================

class CropRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class FertilizerRequest(BaseModel):
    N: float
    P: float
    K: float
    ph: Optional[float] = None
    crop: str = "general"


class YieldRequest(BaseModel):
    temperature: float
    rainfall: float
    ph: float
    N: float
    P: float
    K: float


class IrrigationRequest(BaseModel):
    moisture: float
    ph: float
    city: str


class SensorPayload(BaseModel):
    moisture: float
    temperature: float
    ec: float
    ph: float
    N: Optional[float] = None
    P: Optional[float] = None
    K: Optional[float] = None
    rainfall: Optional[float] = None


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "AgriAI Smart Farming API is running",
        "status": "online",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    """
    Always returns 200 so Render's health check passes while models
    load. Check "ready" to see whether predictions are available.
    """
    return {
        "status": "healthy" if state["ready"] else state["stage"],
        "ready": state["ready"],
        "error": state["error"],
        "crop_model": crop_model is not None,
        "yield_model": yield_model is not None,
        "leaf_model": model is not None,
        "disease_classes": len(classes),
        "soil_ml_model_loaded": soil_ml_model is not None,
        "irrigation_ml_model_loaded": irrigation_ml_model is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# CROP RECOMMENDATION
# ============================================================

@app.get("/predict_crop")
def predict_crop(
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
):
    """
    Crop model feature order:
    [N, P, K, temperature, humidity, ph, rainfall]

    humidity is AIR humidity. It is NOT soil moisture.
    """
    require_ready()

    try:
        data = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=[
                "N", "P", "K", "temperature",
                "humidity", "ph", "rainfall",
            ],
        )

        prediction = crop_model.predict(data)[0]

        return {"recommended_crop": str(prediction)}

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Crop prediction failed: {str(exc)}",
        )


# ============================================================
# FERTILIZER RECOMMENDATION
# ============================================================

@app.post("/fertilizer")
def fertilizer(request: FertilizerRequest):
    try:
        result = recommend_fertilizer(
            request.N,
            request.P,
            request.K,
            request.ph,
        )

        return {"recommendation": result}

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Fertilizer recommendation failed: {str(exc)}",
        )


# ============================================================
# YIELD PREDICTION
# ============================================================

@app.get("/predict_yield")
def predict_yield(
    temperature: float,
    rainfall: float,
    ph: float,
    N: float,
    P: float,
    K: float,
):
    """
    Yield model feature order:
    [temperature, rainfall, ph, N, P, K]
    """
    require_ready()

    try:
        data = pd.DataFrame(
            [[temperature, rainfall, ph, N, P, K]],
            columns=["temperature", "rainfall", "ph", "N", "P", "K"],
        )

        prediction = yield_model.predict(data)[0]
        prediction = max(0, float(prediction))

        return {"predicted_yield": prediction}

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Yield prediction failed: {str(exc)}",
        )


# ============================================================
# DISEASE DETECTION
# ============================================================

@app.post("/predict_disease")
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict plant disease from an uploaded leaf image.

    Model probability: 0.0 - 1.0
    API confidence:    0 - 100 (converted exactly once, here)
    """
    require_ready()

    try:
        if not file.content_type:
            raise HTTPException(
                status_code=400,
                detail="File content type is missing.",
            )

        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Please upload an image file.",
            )

        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        tensor = disease_transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

        class_index = int(predicted.item())

        if class_index < 0 or class_index >= len(classes):
            raise RuntimeError(
                "Predicted class index is outside "
                "the available disease classes."
            )

        disease_name = str(classes[class_index])

        confidence_percent = float(confidence.item() * 100.0)

        cure = get_cure(disease_name, confidence_percent)

        return {
            "disease": disease_name,
            "confidence": round(confidence_percent, 2),
            "cure": cure,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Disease prediction failed: {str(exc)}",
        )


# ============================================================
# IRRIGATION AI
# ============================================================

@app.post("/irrigation_ai")
def irrigation_ai(request: IrrigationRequest):
    """
    Rule-based irrigation logic using soil moisture and
    weather (rainfall, temperature, humidity) for the given city.

    pH is accepted as input but not used by the current rules.
    """
    try:
        moisture = float(request.moisture)
        ph = float(request.ph)
        city = request.city.strip()

        if not city:
            raise HTTPException(status_code=400, detail="City is required.")

        if moisture < 0 or moisture > 100:
            raise HTTPException(
                status_code=400,
                detail="Soil moisture must be between 0 and 100.",
            )

        try:
            temperature, humidity, rainfall = get_weather(city)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except requests.RequestException:
            raise HTTPException(
                status_code=502,
                detail="Weather service is unreachable. Try again shortly.",
            )

        if rainfall > 3:
            irrigation = "OFF"
            reason = "Rainfall is sufficient."

        elif moisture < 30 and temperature > 30:
            irrigation = "ON"
            reason = "Soil is very dry and temperature is high."

        elif humidity > 85:
            irrigation = "OFF"
            reason = "Air humidity is very high."

        elif moisture < 40:
            irrigation = "ON"
            reason = "Soil moisture is below the irrigation threshold."

        else:
            irrigation = "OFF"
            reason = "Soil moisture is sufficient."

        return {
            "irrigation": irrigation,
            "reason": reason,
            "moisture": moisture,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "ph": ph,
            "city": city,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Irrigation prediction failed: {str(exc)}",
        )


# ============================================================
# RECEIVE ESP32 SENSOR DATA
# ============================================================

@app.post("/sensor_data")
def receive_sensor_data(payload: SensorPayload):
    """
    Receive real sensor data from ESP32 / RS485.

    State starts as None and is only populated when an ESP32
    sends actual data. rainfall is optional because the soil
    sensor does not measure rainfall.
    """
    with _sensor_lock:
        _latest_sensor_data.update({
            "moisture": payload.moisture,
            "temperature": payload.temperature,
            "ec": payload.ec,
            "ph": payload.ph,
            "N": payload.N,
            "P": payload.P,
            "K": payload.K,
            "rainfall": payload.rainfall,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": "ESP32 / RS485 soil sensor",
        })

        return dict(_latest_sensor_data)


# ============================================================
# GET LATEST ESP32 SENSOR DATA
# ============================================================

@app.get("/sensor_data")
def get_sensor_data():
    with _sensor_lock:
        return dict(_latest_sensor_data)
