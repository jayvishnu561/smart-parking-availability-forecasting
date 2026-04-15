from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .model_service import ModelService


ZONE_CAPACITY = {
    "A": 120,
    "B": 90,
    "C": 70,
    "D": 55,
}

VEHICLE_OCCUPANCY_FACTOR = {
    "car": 1.0,
    "bike": 0.65,
    "ev": 0.95,
    "truck": 1.25,
}

VEHICLE_BASE_FEE = {
    "car": 40.0,
    "bike": 20.0,
    "ev": 35.0,
    "truck": 70.0,
}


def _time_factor(time_of_day: str) -> float:
    hour = int(time_of_day.split(":", maxsplit=1)[0])
    if 7 <= hour <= 10 or 17 <= hour <= 20:
        return 1.15
    if 0 <= hour <= 5:
        return 0.82
    return 1.0


class ForecastRequest(BaseModel):
    sequence: list[float] = Field(
        ...,
        min_length=3,
        description="Historical occupancy sequence values in range [0, 1]",
    )
    time_of_day: str = Field(..., pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    vehicle_type: str = Field(..., pattern=r"^(car|bike|ev|truck)$")
    zone: str = Field(..., pattern=r"^(A|B|C|D)$")
    horizon: int = Field(default=1, ge=1, le=24)


class ForecastResponse(BaseModel):
    forecast: list[float]
    availability_percent: float
    available_slots: int
    estimated_fee: float
    zone: str
    vehicle_type: str
    requested_time: str
    status: str
    model_input_shape: list[int | None]
    model_path: str


app = FastAPI(
    title="Smart Parking Availability Forecast API",
    version="1.0.0",
    description="LSTM-powered occupancy forecasting for urban parking slots.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = ModelService()


@app.get("/health")
def health() -> dict[str, str | list[int | None]]:
    return {
        "status": "ok",
        "model_path": service.model_path,
        "model_input_shape": service.input_shape,
    }


@app.post("/predict", response_model=ForecastResponse)
def predict(payload: ForecastRequest) -> ForecastResponse:
    try:
        forecast = service.predict(payload.sequence, payload.horizon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    mean_occupancy = sum(forecast) / len(forecast)
    mean_occupancy *= VEHICLE_OCCUPANCY_FACTOR[payload.vehicle_type]
    mean_occupancy *= _time_factor(payload.time_of_day)
    mean_occupancy = min(max(mean_occupancy, 0.0), 1.0)

    availability_percent = round((1 - mean_occupancy) * 100, 2)
    capacity = ZONE_CAPACITY[payload.zone]
    available_slots = max(int(round(capacity * (1 - mean_occupancy))), 0)

    dynamic_fee = VEHICLE_BASE_FEE[payload.vehicle_type] * (1 + mean_occupancy * 0.7)
    estimated_fee = round(dynamic_fee, 2)

    if availability_percent >= 60:
        status = "High availability"
    elif availability_percent >= 35:
        status = "Moderate availability"
    else:
        status = "Low availability"

    return ForecastResponse(
        forecast=forecast,
        availability_percent=availability_percent,
        available_slots=available_slots,
        estimated_fee=estimated_fee,
        zone=payload.zone,
        vehicle_type=payload.vehicle_type,
        requested_time=payload.time_of_day,
        status=status,
        model_input_shape=service.input_shape,
        model_path=service.model_path,
    )