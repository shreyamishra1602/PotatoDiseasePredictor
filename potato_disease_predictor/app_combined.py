#!/usr/bin/env python3
"""
Potato Disease Prediction System
Phosphor icons + Accent UI + PDF Report
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
import pickle
import json
import subprocess
import sys
import tempfile
import requests
import io

st.set_page_config(page_title="Potato Disease Predictor", layout="wide")

# ── External resources ──
st.markdown('<link rel="stylesheet" href="https://unpkg.com/@phosphor-icons/web@2.1.1/src/regular/style.css">', unsafe_allow_html=True)
st.markdown('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">', unsafe_allow_html=True)

# ── Accent CSS ──
st.markdown("""<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { max-width: 920px; padding-top: 1.2rem; }

h1 { font-weight: 700; letter-spacing: -0.02em; }
h2 { font-weight: 600; font-size: 1.4rem; }

.accent { color: #16a34a; }

/* ── Horizontal nav ── */
.nav-bar {
    display: flex; gap: 0; margin-bottom: 1.5rem;
    border-bottom: 2px solid #e5e7eb;
}
.nav-item {
    display: flex; align-items: center; gap: 0.4rem;
    padding: 0.7rem 1.2rem; cursor: pointer;
    font-weight: 500; font-size: 0.9rem; color: #6b7280;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px; transition: all 0.15s;
    text-decoration: none;
}
.nav-item:hover { color: #16a34a; }
.nav-item.active { color: #16a34a; border-bottom-color: #16a34a; font-weight: 600; }
.nav-item i { font-size: 1.1rem; }

/* ── Cards ── */
.result-card {
    padding: 1.5rem; border-radius: 12px;
    text-align: center; margin: 0.5rem 0;
}
.result-early { background: #fef9c3; border: 1px solid #eab308; }
.result-late { background: #fee2e2; border: 1px solid #ef4444; }
.result-healthy { background: #dcfce7; border: 1px solid #22c55e; }
.result-earlyblight { background: #fef9c3; border: 1px solid #eab308; }
.result-lateblight { background: #fee2e2; border: 1px solid #ef4444; }

.prob-row {
    display: flex; align-items: center; gap: 0.5rem;
    margin: 0.4rem 0; font-size: 0.9rem;
}
.prob-bar { height: 8px; border-radius: 4px; background: #e5e7eb; flex: 1; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 4px; }

.tag {
    display: inline-block; padding: 0.2rem 0.6rem;
    border-radius: 12px; font-size: 0.75rem; font-weight: 600;
}
.tag-green { background: #dcfce7; color: #166534; }
.tag-yellow { background: #fef9c3; color: #854d0e; }
.tag-red { background: #fee2e2; color: #991b1b; }

.day-row {
    display: flex; align-items: center; gap: 1rem;
    padding: 0.5rem 0; border-bottom: 1px solid #f3f4f6;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ──
@st.cache_resource
def load_weather_model():
    model_dir = Path("data/models")
    proc_dir = Path("data/processed")
    model = None
    for name in ["best_weather_model.pkl", "xgboost_weather_model.pkl", "random_forest_weather_model.pkl"]:
        p = model_dir / name
        if p.exists():
            with open(p, "rb") as f:
                model = pickle.load(f)
            break
    le = scaler = features = None
    if (proc_dir / "label_encoder.pkl").exists():
        with open(proc_dir / "label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        with open(proc_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(proc_dir / "feature_names.pkl", "rb") as f:
            features = pickle.load(f)
    return model, le, scaler, features


DISEASE_INFO = {
    "Potato___Early_blight": {
        "name": "Early Blight", "cause": "Alternaria solani (fungus)",
        "symptoms": "Dark concentric rings (target spots) on older leaves, yellowing",
        "action": "Apply mancozeb/chlorothalonil fungicide. Remove infected leaves. Rotate crops.",
    },
    "Potato___Late_blight": {
        "name": "Late Blight", "cause": "Phytophthora infestans",
        "symptoms": "Water-soaked dark patches, white fuzz on undersides. Spreads fast in cool wet weather.",
        "action": "Apply metalaxyl fungicide IMMEDIATELY. Destroy infected plants. Stop overhead watering.",
    },
    "Potato___healthy": {
        "name": "Healthy", "cause": "No disease",
        "symptoms": "Uniform green, no spots or discoloration.",
        "action": "Continue regular monitoring and crop hygiene.",
    },
}

CITY_COORDS = {
    "Delhi": (28.6139, 77.2090), "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946), "Kolkata": (22.5726, 88.3639),
    "Agra": (27.1767, 78.0081), "Shimla": (31.1048, 77.1734),
    "Pune": (18.5204, 73.8567), "Chandigarh": (30.7333, 76.7794),
    "Lucknow": (26.8467, 80.9462), "Jaipur": (26.9124, 75.7873),
    "Patna": (25.6093, 85.1376), "Bhopal": (23.2599, 77.4126),
    "Custom": None,
}


def predict_weather(model, scaler, le, feature_names, inputs):
    vec = np.zeros((1, len(feature_names)))
    for i, f in enumerate(feature_names):
        if f in inputs:
            vec[0, i] = inputs[f]
        elif f == "temp_humidity_interaction":
            vec[0, i] = inputs.get("temperature", 0) * inputs.get("humidity", 0) / 100
        elif f == "is_blight_favorable":
            t, h = inputs.get("temperature", 0), inputs.get("humidity", 0)
            vec[0, i] = 1.0 if (15 <= t <= 25 and h >= 85) else 0.0
        elif f == "temp_dewpoint_diff":
            vec[0, i] = inputs.get("temperature", 0) - inputs.get("dew_point", 0)
        elif f == "low_pressure":
            vec[0, i] = 1.0 if inputs.get("pressure", 1013) < 1010 else 0.0
    scaled = scaler.transform(vec)
    pred = model.predict(scaled)
    probs = model.predict_proba(scaled)[0] if hasattr(model, "predict_proba") else None
    return le.classes_[pred[0]], probs


def fetch_weather_history(lat, lon, days=7):
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": "temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_max,surface_pressure_mean,visibility_mean,wind_direction_10m_dominant",
        "timezone": "auto",
    }, timeout=10)
    resp.raise_for_status()
    data = resp.json()["daily"]
    rows = []
    for i, d in enumerate(data["time"]):
        rows.append({
            "date": d,
            "temperature": round(data["temperature_2m_mean"][i] or 20, 1),
            "humidity": round(data["relative_humidity_2m_mean"][i] or 60, 1),
            "wind_speed": round(data["wind_speed_10m_max"][i] or 10, 1),
            "wind_bearing": round(data["wind_direction_10m_dominant"][i] or 180, 1),
            "pressure": round(data["surface_pressure_mean"][i] or 1013, 1),
            "visibility": round((data["visibility_mean"][i] or 10000) / 1000, 1),
        })
    return pd.DataFrame(rows)


def run_vit_inference(image_path):
    result = subprocess.run(
        [sys.executable, "predict_image.py", str(image_path)],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout.strip())


def generate_pdf_report(city, predictions, df_pred):
    """Generate a PDF report using only standard library"""
    from datetime import datetime

    lines = []
    lines.append("POTATO BLIGHT RISK REPORT")
    lines.append("=" * 50)
    lines.append(f"Location: {city}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Period: {predictions[0]['date']} to {predictions[-1]['date']}")
    lines.append("")

    late_count = sum(1 for p in predictions if "late" in p["prediction"].lower())
    early_count = sum(1 for p in predictions if "early" in p["prediction"].lower())
    lines.append(f"SUMMARY")
    lines.append(f"  Days analyzed:     {len(predictions)}")
    lines.append(f"  Early Blight days: {early_count}")
    lines.append(f"  Late Blight days:  {late_count}")
    lines.append("")

    if late_count > early_count:
        lines.append("RISK LEVEL: HIGH - Late Blight dominant")
        lines.append("ACTION: Apply metalaxyl fungicide immediately.")
    elif early_count > 0:
        lines.append("RISK LEVEL: MODERATE - Early Blight present")
        lines.append("ACTION: Apply mancozeb/chlorothalonil fungicide.")
    else:
        lines.append("RISK LEVEL: LOW")
    lines.append("")

    lines.append("DAY-BY-DAY ANALYSIS")
    lines.append("-" * 50)
    lines.append(f"{'Date':<12} {'Prediction':<16} {'Temp':<8} {'Humidity':<10} {'Wind':<8}")
    lines.append("-" * 50)
    for p in predictions:
        lines.append(f"{p['date']:<12} {p['prediction']:<16} {p['temp']}C{'':<4} {p['humidity']}%{'':<5} {p['wind']} km/h")
    lines.append("")

    lines.append("DISEASE INFORMATION")
    lines.append("-" * 50)
    lines.append("Early Blight (Alternaria solani)")
    lines.append("  - Dark concentric rings on older leaves")
    lines.append("  - Treat with mancozeb/chlorothalonil")
    lines.append("")
    lines.append("Late Blight (Phytophthora infestans)")
    lines.append("  - Water-soaked dark patches, white fuzz")
    lines.append("  - URGENT: Apply metalaxyl, destroy infected plants")
    lines.append("")
    lines.append("=" * 50)
    lines.append("Generated by Potato Disease Prediction System")
    lines.append("Weather data: Open-Meteo API | Model: XGBoost (97% accuracy)")

    return "\n".join(lines)


# ── Navigation ──
w_model, w_le, w_scaler, w_features = load_weather_model()
vit_ready = Path("data/models/vit_best.pth").exists() or Path("data/models/vit_final.pth").exists()

# Navigation via pills (horizontal)
nav_items = ["Live Weather", "Manual", "Leaf Scan", "Report", "About"]
page = st.pills("Navigation", nav_items, default="Live Weather", label_visibility="collapsed")

# Model status in sidebar
with st.sidebar:
    st.markdown("### Status")
    st.markdown(f'<i class="ph ph-cloud-sun"></i> Weather: <span class="tag tag-green">ready</span>' if w_model else
                f'<i class="ph ph-cloud-sun"></i> Weather: <span class="tag tag-red">missing</span>', unsafe_allow_html=True)
    st.markdown(f'<i class="ph ph-eye"></i> ViT: <span class="tag tag-green">ready</span>' if vit_ready else
                f'<i class="ph ph-eye"></i> ViT: <span class="tag tag-yellow">not trained</span>', unsafe_allow_html=True)


# ═══════════════════════════════════════
# Live Weather
# ═══════════════════════════════════════
if page == "Live Weather":
    st.markdown('# <i class="ph ph-cloud-sun"></i> <span class="accent">Potato</span> Blight — Live Weather', unsafe_allow_html=True)
    st.caption("Fetches real weather from the past week and predicts blight risk daily")

    c1, c2 = st.columns([2, 1])
    with c1:
        city = st.selectbox("Location", list(CITY_COORDS.keys()))
    with c2:
        days = st.selectbox("Days", [7, 10, 14], index=0)

    if city == "Custom":
        cc1, cc2 = st.columns(2)
        with cc1:
            lat = st.number_input("Latitude", -90.0, 90.0, 28.6)
        with cc2:
            lon = st.number_input("Longitude", -180.0, 180.0, 77.2)
    else:
        lat, lon = CITY_COORDS[city]

    if st.button("Fetch & Predict", type="primary") and w_model:
        with st.spinner(f"Fetching weather for {city}..."):
            try:
                df_weather = fetch_weather_history(lat, lon, days)
            except Exception as e:
                st.error(f"Failed to fetch weather: {e}")
                st.stop()

        predictions = []
        for _, row in df_weather.iterrows():
            label, probs = predict_weather(w_model, w_scaler, w_le, w_features, row.to_dict())
            prob_dict = {}
            if probs is not None:
                for i, cls in enumerate(w_le.classes_):
                    prob_dict[cls] = round(probs[i] * 100, 1)
            predictions.append({"date": row["date"], "prediction": label, **prob_dict,
                                "temp": row["temperature"], "humidity": row["humidity"],
                                "wind": row["wind_speed"], "pressure": row["pressure"]})

        df_pred = pd.DataFrame(predictions)
        st.session_state["last_predictions"] = predictions
        st.session_state["last_city"] = city
        st.session_state["last_df_pred"] = df_pred

        late_count = sum(1 for p in predictions if "late" in p["prediction"].lower())
        early_count = sum(1 for p in predictions if "early" in p["prediction"].lower())

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Days Analyzed", len(predictions))
        with m2:
            st.metric("Early Blight Days", early_count)
        with m3:
            st.metric("Late Blight Days", late_count)

        for p in predictions:
            is_late = "late" in p["prediction"].lower()
            tag_cls = "tag-red" if is_late else "tag-yellow"
            st.markdown(f"""<div class="day-row">
                <span style="width:90px; font-weight:500;">{p['date']}</span>
                <span class="tag {tag_cls}">{p['prediction']}</span>
                <span style="color:#6b7280; font-size:0.85rem;">{p['temp']}C &middot; {p['humidity']}% hum &middot; {p['wind']} km/h</span>
            </div>""", unsafe_allow_html=True)

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_pred["date"], y=df_pred["temp"], name="Temp (C)",
                                 line=dict(color="#ef4444", width=2)))
        fig.add_trace(go.Scatter(x=df_pred["date"], y=df_pred["humidity"], name="Humidity (%)",
                                 line=dict(color="#3b82f6", width=2), yaxis="y2"))
        fig.update_layout(height=280, margin=dict(l=0, r=40, t=10, b=0),
                          yaxis=dict(title="Temp (C)"),
                          yaxis2=dict(title="Humidity (%)", overlaying="y", side="right"),
                          legend=dict(x=0, y=1.15, orientation="h"))
        st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════
# Manual Prediction
# ═══════════════════════════════════════
elif page == "Manual":
    st.markdown('# <i class="ph ph-sliders-horizontal"></i> <span class="accent">Potato</span> Manual Prediction', unsafe_allow_html=True)
    st.caption("Enter weather conditions manually")

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        temp = st.slider("Temperature (C)", 5.0, 40.0, 20.0, 0.5)
        humidity = st.slider("Humidity (%)", 20.0, 100.0, 75.0, 1.0)
        wind = st.slider("Wind Speed (km/h)", 0.0, 30.0, 10.0, 0.5)
        pc1, pc2 = st.columns(2)
        with pc1:
            pressure = st.number_input("Pressure (hPa)", 980, 1050, 1013)
        with pc2:
            visibility = st.number_input("Visibility (km)", 0.0, 20.0, 10.0, 0.5)
        go_btn = st.button("Predict", type="primary")

    with c2:
        if go_btn and w_model:
            inputs = {"temperature": temp, "humidity": humidity, "wind_speed": wind,
                      "pressure": float(pressure), "visibility": visibility, "wind_bearing": 180.0}
            label, probs = predict_weather(w_model, w_scaler, w_le, w_features, inputs)
            css = "result-lateblight" if "late" in label.lower() else (
                "result-earlyblight" if "early" in label.lower() else "result-healthy")
            st.markdown(f'<div class="result-card {css}">'
                        f'<div style="font-size:1.6rem;font-weight:700;">{label}</div>'
                        f'<div style="font-size:0.85rem;color:#6b7280;margin-top:4px;">XGBoost prediction</div>'
                        f'</div>', unsafe_allow_html=True)
            if probs is not None:
                st.markdown("##### Confidence")
                for i, cls in enumerate(w_le.classes_):
                    pct = probs[i] * 100
                    color = "#ef4444" if "late" in cls.lower() else ("#eab308" if "early" in cls.lower() else "#22c55e")
                    st.markdown(f'<div class="prob-row"><span style="width:100px;">{cls}</span>'
                                f'<div class="prob-bar"><div class="prob-fill" style="width:{pct}%;background:{color};"></div></div>'
                                f'<span style="width:50px;text-align:right;">{pct:.1f}%</span></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════
# Leaf Scan
# ═══════════════════════════════════════
elif page == "Leaf Scan":
    st.markdown('# <i class="ph ph-leaf"></i> <span class="accent">Potato</span> Leaf Scan', unsafe_allow_html=True)
    st.caption("Upload a potato leaf photo for instant ViT disease detection")

    if not vit_ready:
        st.info("ViT model not trained yet. Run `python scripts/5_train_vit_model.py` first.")

    uploaded = st.file_uploader("Drop a leaf image here", type=["jpg", "jpeg", "png"])
    if uploaded:
        from PIL import Image
        img = Image.open(uploaded).convert("RGB")
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.image(img, caption="Uploaded leaf", width=350)
        with c2:
            if vit_ready:
                with st.spinner("Analyzing..."):
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        img.save(tmp.name)
                        result = run_vit_inference(tmp.name)
                if result:
                    pred = result["_prediction"]
                    conf = result["_confidence"]
                    info = DISEASE_INFO.get(pred, DISEASE_INFO["Potato___healthy"])
                    css = "result-late" if "late" in pred.lower() else ("result-early" if "early" in pred.lower() else "result-healthy")
                    st.markdown(f'<div class="result-card {css}">'
                                f'<div style="font-size:1.6rem;font-weight:700;">{info["name"]}</div>'
                                f'<div style="font-size:0.95rem;margin-top:4px;">{conf:.1f}% confidence</div>'
                                f'</div>', unsafe_allow_html=True)
                    for cls_key in ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]:
                        if cls_key in result:
                            pct = result[cls_key]
                            name = DISEASE_INFO[cls_key]["name"]
                            color = "#ef4444" if "late" in cls_key.lower() else ("#eab308" if "early" in cls_key.lower() else "#22c55e")
                            st.markdown(f'<div class="prob-row"><span style="width:100px;">{name}</span>'
                                        f'<div class="prob-bar"><div class="prob-fill" style="width:{pct}%;background:{color};"></div></div>'
                                        f'<span style="width:50px;text-align:right;">{pct:.1f}%</span></div>', unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown(f"**Cause:** {info['cause']}")
                    st.markdown(f"**Symptoms:** {info['symptoms']}")
                    st.markdown(f"**Action:** {info['action']}")


# ═══════════════════════════════════════
# PDF Report
# ═══════════════════════════════════════
elif page == "Report":
    st.markdown('# <i class="ph ph-file-pdf"></i> <span class="accent">Potato</span> Blight Report', unsafe_allow_html=True)
    st.caption("Download a PDF-style report of the last weather analysis")

    if "last_predictions" in st.session_state:
        predictions = st.session_state["last_predictions"]
        city = st.session_state["last_city"]
        df_pred = st.session_state["last_df_pred"]

        st.success(f"Report ready for **{city}** ({len(predictions)} days)")

        report_text = generate_pdf_report(city, predictions, df_pred)
        st.text(report_text)

        st.download_button(
            label="Download Report (.txt)",
            data=report_text,
            file_name=f"potato_blight_report_{city.lower()}_{date.today().isoformat()}.txt",
            mime="text/plain",
        )

    else:
        st.info("No analysis data yet. Go to **Live Weather**, run a prediction first, then come back here to download the report.")


# ═══════════════════════════════════════
# About
# ═══════════════════════════════════════
elif page == "About":
    st.markdown('# <i class="ph ph-info"></i> <span class="accent">About</span>', unsafe_allow_html=True)
    st.markdown("""
    ### Potato Disease Prediction System

    **Weather Prediction** — XGBoost model trained on 4,020 weather records.
    Predicts Early Blight vs Late Blight from temperature, humidity, wind, pressure, visibility.
    Accuracy: **97.01%**

    **Leaf Scan** — Vision Transformer (ViT-Small) fine-tuned on 2,152 potato leaf images.
    Classes: Early Blight, Late Blight, Healthy. Accuracy: **100%**

    **Live Weather** — Open-Meteo free API (no key needed). Fetches real historical weather
    for any Indian city and predicts blight risk for each day.

    ---

    **Stack:** PyTorch + timm (ViT) / scikit-learn + XGBoost / Streamlit / Open-Meteo API

    Optimized for MacBook with Apple Silicon (MPS).
    """)
