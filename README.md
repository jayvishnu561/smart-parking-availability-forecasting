# Smart Parking Availability Forecasting

Deep-learning project to forecast parking occupancy and estimate slot availability using an LSTM model.

## Project Structure

- `lstm_parking_model.keras` - Trained LSTM model file
- `backend/` - FastAPI inference service
- `frontend/` - React (Vite) interactive dashboard

## Features

- Uses historical occupancy sequence input (0 to 1 values)
- Forecasts future occupancy from your trained LSTM model
- Converts occupancy into parking availability percentage
- Interactive web dashboard for city authorities and commuters

## Backend Setup (FastAPI)

1. Open terminal in project root.
2. Create and activate virtual environment.
3. Install dependencies:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

4. Run API:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`

## Frontend Setup (React + Vite)

1. Open a new terminal:

```powershell
cd frontend
npm install
npm run dev
```

2. Open: `http://localhost:5173`

## Optional Environment Variable

If model path changes, set:

```powershell
$env:MODEL_PATH="C:\full\path\to\lstm_parking_model.keras"
```

## API Example

POST `/predict`

```json
{
  "sequence": [0.72, 0.68, 0.61, 0.59, 0.64, 0.69],
  "horizon": 4
}
```

Response includes forecast values, availability percentage, and demand status.