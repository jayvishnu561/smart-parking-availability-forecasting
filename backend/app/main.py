from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .history_service import HistoryService
from .model_service import ModelService


ZONE_CAPACITY = {
    "mall": 120,
    "commercial": 90,
    "office": 70,
    "residential": 55,
    "hospital": 85,
}


class ForecastRequest(BaseModel):
    sequence: list[float] | None = Field(
        default=None,
        description="Optional historical occupancy sequence values in range [0, 1]",
    )
    # Accept 12-hour clock like "12:34 PM" (case-insensitive, optional space)
    time_of_day: str = Field(..., pattern=r"^(?i)(0?[1-9]|1[0-2]):[0-5]\d\s?(AM|PM)$")
    vehicle_type: str = Field(..., pattern=r"^(car|bike|ev|truck)$")
    zone: str = Field(..., pattern=r"^(mall|commercial|office|residential|hospital)$")
    horizon: int = Field(default=1, ge=1, le=24)


class ForecastResponse(BaseModel):
    forecast: list[float]
    availability_percent: float
    available_slots: int
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
history_service = HistoryService()


@app.get("/health")
def health() -> dict[str, str | list[int | None]]:
    return {
        "status": "ok",
        "model_path": service.model_path,
        "model_input_shape": service.input_shape,
    }


@app.post("/predict", response_model=ForecastResponse)
def predict(payload: ForecastRequest) -> ForecastResponse:
    if payload.sequence is not None and len(payload.sequence) < 3:
        raise HTTPException(status_code=400, detail="Sequence length must be at least 3 values.")

    # Normalize 12-hour time string (e.g. "12:30 PM") to 24-hour "HH:MM" for internal use
    from datetime import datetime

    original_time = payload.time_of_day
    try:
        dt = datetime.strptime(payload.time_of_day.strip().upper(), "%I:%M %p")
        normalized_time = dt.strftime("%H:%M")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid time_of_day format: {payload.time_of_day}") from exc

    sequence = payload.sequence
    if sequence is None:
        required_length = service.required_sequence_length()
        try:
            sequence = history_service.get_sequence(
                zone=payload.zone,
                vehicle_type=payload.vehicle_type,
                time_of_day=normalized_time,
                required_length=required_length,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        forecast = service.predict(sequence, payload.horizon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    mean_occupancy = sum(forecast) / len(forecast)
    mean_occupancy = min(max(mean_occupancy, 0.0), 1.0)

    availability_percent = round((1 - mean_occupancy) * 100, 2)
    capacity = ZONE_CAPACITY[payload.zone]
    available_slots = max(int(round(capacity * (1 - mean_occupancy))), 0)

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
        zone=payload.zone,
        vehicle_type=payload.vehicle_type,
        requested_time=original_time,
        status=status,
        model_input_shape=service.input_shape,
        model_path=service.model_path,
    )