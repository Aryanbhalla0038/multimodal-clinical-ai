import { useState } from "react";
import { PredictionBar } from "./components/PredictionBar";
import { VitalsChart } from "./components/VitalsChart";
import { Upload, FileText, Activity, AlertTriangle, Loader2 } from "lucide-react";

const API = "/api";

interface Result {
  predictions: Record<string, number>;
  gradcam_image: string | null;
  gradcam_label: string | null;
  vitals_importance: number[] | null;
  missing_modalities: string[];
  disclaimer: string;
}

export default function App() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [note, setNote] = useState<string>("");
  const [vitalsFile, setVitalsFile] = useState<File | null>(null);
  const [gradcamLabel, setGradcamLabel] = useState<string>("");
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canRun = imageFile || note.trim() || vitalsFile;

  async function runPrediction() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const fd = new FormData();
      if (imageFile) fd.append("image", imageFile);
      if (note.trim()) fd.append("note", note);
      if (vitalsFile) fd.append("vitals_csv", vitalsFile);
      if (gradcamLabel) fd.append("gradcam_label", gradcamLabel);

      const res = await fetch(`${API}/predict`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: Result = await res.json();
      setResult(data);
      if (!gradcamLabel && data.gradcam_label) setGradcamLabel(data.gradcam_label);
    } catch (err: any) {
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  const sortedPredictions = result
    ? Object.entries(result.predictions).sort(([, a], [, b]) => b - a)
    : [];

  return (
    <div style={{ minHeight: "100vh", paddingBottom: 50 }}>
      <header style={{
        background: "var(--brand-primary)", color: "white",
        padding: "20px 32px", display: "flex", alignItems: "center", gap: 16,
      }}>
        <Activity size={28}/>
        <div>
          <h1 style={{ margin: 0, fontSize: 22 }}>Multimodal Clinical AI</h1>
          <div style={{ fontSize: 12, opacity: 0.85 }}>
            ViT + BioBERT + LSTM · 14-label CheXpert diagnosis
          </div>
        </div>
      </header>

      <main style={{ maxWidth: 1200, margin: "24px auto", padding: "0 24px",
                     display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        {/* ----- LEFT: INPUT ----- */}
        <section>
          <div className="card">
            <h2 style={{ marginTop: 0 }}><Upload size={18}/> Chest X-ray</h2>
            <input
              type="file" accept="image/*,.dcm"
              onChange={(e) => setImageFile(e.target.files?.[0] ?? null)}
            />
            {imageFile && <div style={{ fontSize: 12, marginTop: 6, color: "var(--text-muted)" }}>
              {imageFile.name}
            </div>}
          </div>

          <div className="card">
            <h2 style={{ marginTop: 0 }}><FileText size={18}/> Clinical note</h2>
            <textarea
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="Paste discharge summary, radiology report, or Assessment & Plan..."
              rows={8}
              style={{ width: "100%", padding: 10, borderRadius: 6,
                       border: "1px solid var(--border)", resize: "vertical" }}
            />
          </div>

          <div className="card">
            <h2 style={{ marginTop: 0 }}><Activity size={18}/> Vitals CSV</h2>
            <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 8 }}>
              Columns: charttime, heart_rate, sbp, dbp, spo2, resp_rate, temp
            </div>
            <input
              type="file" accept=".csv"
              onChange={(e) => setVitalsFile(e.target.files?.[0] ?? null)}
            />
            {vitalsFile && <div style={{ fontSize: 12, marginTop: 6, color: "var(--text-muted)" }}>
              {vitalsFile.name}
            </div>}
          </div>

          <button className="btn-primary" disabled={!canRun || loading} onClick={runPrediction}>
            {loading ? <><Loader2 size={14} style={{ verticalAlign: "middle",
              animation: "spin 1s linear infinite", marginRight: 6 }}/>Analyzing...</>
              : "Run Analysis"}
          </button>
          {error && <div style={{ color: "var(--severity-high)", marginTop: 10 }}>{error}</div>}
        </section>

        {/* ----- RIGHT: RESULTS ----- */}
        <section>
          {result?.missing_modalities && result.missing_modalities.length > 0 && (
            <div className="card" style={{ background: "#FFF8E1", borderColor: "#F0B100" }}>
              <AlertTriangle size={16} color="#9a6700"/>
              <span style={{ marginLeft: 8, fontSize: 13, color: "#9a6700" }}>
                Prediction accuracy may be reduced — missing: {result.missing_modalities.join(", ")}.
              </span>
            </div>
          )}

          {result && (
            <div className="card">
              <h3 style={{ marginTop: 0 }}>Predictions (14 CheXpert labels)</h3>
              {sortedPredictions.map(([label, prob]) => (
                <PredictionBar key={label} label={label} probability={prob}/>
              ))}
            </div>
          )}

          {result?.gradcam_image && (
            <div className="card">
              <h3 style={{ marginTop: 0 }}>Grad-CAM · {result.gradcam_label}</h3>
              <select
                value={gradcamLabel}
                onChange={(e) => setGradcamLabel(e.target.value)}
                style={{ marginBottom: 10, padding: 6, borderRadius: 4,
                         border: "1px solid var(--border)" }}
              >
                {Object.keys(result.predictions).map((l) => (
                  <option key={l} value={l}>{l}</option>
                ))}
              </select>
              <button
                className="btn-primary"
                style={{ marginLeft: 8, padding: "6px 12px" }}
                onClick={runPrediction}
                disabled={loading}
              >Refresh</button>
              <div>
                <img src={result.gradcam_image} alt="Grad-CAM overlay"
                     style={{ width: "100%", borderRadius: 6, marginTop: 8 }}/>
              </div>
            </div>
          )}

          <VitalsChart importance={result?.vitals_importance ?? null}/>

          {!result && !loading && (
            <div className="card" style={{ color: "var(--text-muted)" }}>
              Provide at least one modality and click <b>Run Analysis</b>.
            </div>
          )}
        </section>
      </main>

      <div className="footer-disclaimer">
        For research purposes only. Not for clinical decision-making.
      </div>

      <style>{`@keyframes spin { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }`}</style>
    </div>
  );
}
