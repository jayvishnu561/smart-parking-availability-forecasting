import { useState } from "react";
import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const defaultHorizon = 4;

const ZONE_NAMES = {
  mall: "Mall",
  commercial: "Commercial",
  office: "Office",
  residential: "Residential",
  hospital: "Hospital"
};

export default function App() {
  const [timeOfDay, setTimeOfDay] = useState("09:00");
  const [vehicleType, setVehicleType] = useState("car");
  const [zone, setZone] = useState("mall");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const runPrediction = async () => {
    setLoading(true);
    setError("");

    try {
      const to12Hour = (time24) => {
        if (!time24) return time24;
        const [hh, mm] = time24.split(":");
        let hour = parseInt(hh, 10);
        const ampm = hour >= 12 ? "PM" : "AM";
        hour = hour % 12 || 12;
        return `${hour.toString().padStart(2, "0")}:${mm} ${ampm}`;
      };

      const response = await axios.post(`${API_URL}/predict`, {
        time_of_day: to12Hour(timeOfDay),
        vehicle_type: vehicleType,
        zone,
        horizon: defaultHorizon,
      });
      setResult(response.data);
    } catch (err) {
      const detail = err?.response?.data?.detail;
      let errorMessage = err.message || "Prediction request failed. Check backend server.";
      if (typeof detail === "string") {
        errorMessage = detail;
      } else if (Array.isArray(detail) && detail.length > 0) {
        errorMessage = detail.map((item) => item?.msg).filter(Boolean).join("; ") || errorMessage;
      }
      setError(errorMessage);
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
          Enter time, vehicle type, and zone to estimate available slots powered by your
          trained LSTM model.
        </p>
      </header>

      <main className="layout">
        <section className="card control-panel">
          <h2>Prediction Inputs</h2>
          <p className="helper-text">History is selected dynamically from stored records for the selected context.</p>

          <div className="input-grid">
            <div>
              <label htmlFor="timeOfDay">Time</label>
              <input
                id="timeOfDay"
                type="time"
                step={3600}
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
                <option value="hospital">Hospital</option>
                <option value="office">Office</option>
                <option value="residential">Residential</option>
                <option value="mall">Mall</option>
                <option value="commercial">Commercial</option>
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
                setTimeOfDay("09:00");
                setVehicleType("car");
                setZone("mall");
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
                <span>Available Slots</span>
                <strong>{result.available_slots}</strong>
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
