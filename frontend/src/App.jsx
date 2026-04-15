import { useState } from "react";
import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const seededSequence = [0.72, 0.68, 0.61, 0.59, 0.64, 0.69, 0.75, 0.77, 0.7, 0.62, 0.56, 0.51];
const defaultHorizon = 4;

const ZONE_NAMES = {
  A: "City Mall",
  B: "Grand Theater",
  C: "Central Stadium",
  D: "Metro Station"
};

export default function App() {
  const [timeOfDay, setTimeOfDay] = useState("09:30");
  const [vehicleType, setVehicleType] = useState("car");
  const [zone, setZone] = useState("A");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const runPrediction = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await axios.post(`${API_URL}/predict`, {
        sequence: seededSequence,
        time_of_day: timeOfDay,
        vehicle_type: vehicleType,
        zone,
        horizon: defaultHorizon
      });
      setResult(response.data);
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Prediction request failed. Check backend server.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-shell">
      <header className="hero">
        <p className="kicker">Smart Mobility Intelligence</p>
        <h1>Find Parking Faster With Live Availability Forecasts</h1>
        <p>
          Enter time, vehicle type, and zone to estimate available slots and dynamic parking fee powered by your
          trained LSTM model.
        </p>
      </header>

      <main className="layout">
        <section className="card control-panel">
          <h2>Prediction Inputs</h2>
          <p className="helper-text">Only these three fields are needed from users.</p>

          <div className="input-grid">
            <div>
              <label htmlFor="timeOfDay">Time</label>
              <input
                id="timeOfDay"
                type="time"
                value={timeOfDay}
                onChange={(e) => setTimeOfDay(e.target.value)}
              />
            </div>

            <div>
              <label htmlFor="vehicleType">Vehicle Type</label>
              <select
                id="vehicleType"
                value={vehicleType}
                onChange={(e) => setVehicleType(e.target.value)}
              >
                <option value="car">Car</option>
                <option value="bike">Bike</option>
                <option value="ev">EV</option>
                <option value="truck">Truck</option>
              </select>
            </div>

            <div>
              <label htmlFor="zone">Parking Zone</label>
              <select id="zone" value={zone} onChange={(e) => setZone(e.target.value)}>
                <option value="A">City Mall</option>
                <option value="B">Grand Theater</option>
                <option value="C">Central Stadium</option>
                <option value="D">Metro Station</option>
              </select>
            </div>
          </div>

          <div className="button-row">
            <button onClick={runPrediction} disabled={loading}>
              {loading ? "Running model..." : "Run Forecast"}
            </button>
            <button
              className="ghost"
              onClick={() => {
                setTimeOfDay("09:30");
                setVehicleType("car");
                setZone("A");
              }}
            >
              Reset Sample
            </button>
          </div>

          {error && <p className="error-text">{error}</p>}
        </section>

        <section className="card outcome-panel">
          <h2>Forecast Outcome</h2>
          {result ? (
            <div className="status-grid">
              <article>
                <span>Availability</span>
                <strong>{result.availability_percent}%</strong>
              </article>
              <article>
                <span>Available Slots</span>
                <strong>{result.available_slots}</strong>
              </article>
              <article>
                <span>Estimated Fee</span>
                <strong>Rs {result.estimated_fee}</strong>
              </article>
              <article>
                <span>Demand Signal</span>
                <strong>{result.status}</strong>
              </article>
              <article>
                <span>Request Context</span>
                <strong>
                  {ZONE_NAMES[result.zone] ?? result.zone} • {result.vehicle_type.toUpperCase()} • {result.requested_time}
                </strong>
              </article>
            </div>
          ) : (
            <p className="empty-hint">Run a forecast to view occupancy predictions and parking availability signal.</p>
          )}
        </section>
      </main>

      <footer>
        Target users: city authorities and urban commuters. Impact: lower emissions and smoother traffic flow.
      </footer>
    </div>
  );
}
