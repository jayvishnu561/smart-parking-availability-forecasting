from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


VEHICLE_TO_DATASET = {
    "car": "Car",
    "bike": "Bike",
    "ev": "Electric",
    "truck": "SUV",
}


@dataclass(frozen=True)
class HistoryPoint:
    timestamp: datetime
    occupancy_ratio: float


class HistoryService:
    def __init__(self) -> None:
        root = Path(__file__).resolve().parents[2]
        self.dataset_path = root / "smart_parking_usage_occupancy_analytics.csv"

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        self._rows = self._load_rows()

    def _load_rows(self) -> list[dict[str, str]]:
        with self.dataset_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader)

    @staticmethod
    def _clip_ratio(value: float) -> float:
        return min(max(value, 0.0), 1.0)

    def _select_rows(self, zone: str, vehicle_type: str) -> list[HistoryPoint]:
        dataset_zone = zone.capitalize()
        dataset_vehicle = VEHICLE_TO_DATASET[vehicle_type]
        history: list[HistoryPoint] = []

        for row in self._rows:
            if row.get("zone") != dataset_zone:
                continue
            if row.get("vehicle_type") != dataset_vehicle:
                continue

            total_slots = float(row["total_slots"])
            occupied_slots = float(row["occupied_slots"])
            if total_slots <= 0:
                continue

            ratio = self._clip_ratio(occupied_slots / total_slots)
            timestamp = datetime.strptime(row["date_time"], "%Y-%m-%d %H:%M:%S")
            history.append(HistoryPoint(timestamp=timestamp, occupancy_ratio=ratio))

        history.sort(key=lambda item: item.timestamp)
        return history

    @staticmethod
    def _to_minutes(time_of_day: str) -> int:
        hour, minute = time_of_day.split(":", maxsplit=1)
        return int(hour) * 60 + int(minute)

    @staticmethod
    def _point_minutes(point: HistoryPoint) -> int:
        return point.timestamp.hour * 60 + point.timestamp.minute

    @staticmethod
    def _minute_distance(a: int, b: int) -> int:
        direct = abs(a - b)
        return min(direct, 1440 - direct)

    def get_sequence(
        self,
        *,
        zone: str,
        vehicle_type: str,
        time_of_day: str,
        required_length: int,
    ) -> list[float]:
        if required_length <= 0:
            raise ValueError("Required sequence length must be positive.")

        history = self._select_rows(zone, vehicle_type)
        if len(history) == 0:
            raise ValueError(
                f"No historical records found for zone={zone} and vehicle_type={vehicle_type}."
            )

        target_minutes = self._to_minutes(time_of_day)
        matching_indices = [
            idx
            for idx, point in enumerate(history)
            if self._point_minutes(point) == target_minutes
        ]

        if matching_indices:
            end_idx = matching_indices[-1] + 1
        else:
            available_minutes = sorted({self._point_minutes(point) for point in history})
            nearest_minutes = min(
                available_minutes,
                key=lambda candidate: self._minute_distance(target_minutes, candidate),
            )
            nearest_indices = [
                idx for idx, point in enumerate(history) if self._point_minutes(point) == nearest_minutes
            ]
            end_idx = nearest_indices[-1] + 1

        start_idx = max(0, end_idx - required_length)
        selected = [point.occupancy_ratio for point in history[start_idx:end_idx]]

        if len(selected) < required_length:
            pad_value = selected[0] if selected else history[0].occupancy_ratio
            selected = [pad_value] * (required_length - len(selected)) + selected

        return [round(float(v), 4) for v in selected]
