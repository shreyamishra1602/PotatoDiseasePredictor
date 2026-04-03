import streamlit as st
import time
import random
import plotly.graph_objects as go
from datetime import datetime
import math

st.set_page_config(page_title="Smart Garden IoT", page_icon="🌱", layout="wide")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ── Plant config ──
PLANTS = [
    {"id": "tomato", "name": "Tomato",       "emoji": "🍅", "drain": 3.0, "threshold": 35, "start": 72},
    {"id": "tulsi",  "name": "Tulsi",        "emoji": "🌿", "drain": 2.0, "threshold": 25, "start": 60},
    {"id": "cactus", "name": "Cactus",       "emoji": "🌵", "drain": 0.8, "threshold": 10, "start": 40},
    {"id": "rose",   "name": "Rose",         "emoji": "🌹", "drain": 2.8, "threshold": 40, "start": 78},
    {"id": "mint",   "name": "Mint",         "emoji": "🌱", "drain": 3.2, "threshold": 38, "start": 68},
    {"id": "chili",  "name": "Chili Pepper", "emoji": "🌶️", "drain": 2.5, "threshold": 30, "start": 55},
]
COLORS = ["#e05d44", "#2e9e6e", "#e8963e", "#c44569", "#7c5cbf", "#1fa89c"]

# ── Theme ──
DK = st.session_state.dark_mode
if DK:
    T = dict(bg="#111318", card="#1a1d24", border="#2a2d35", text="#e8e6e1", muted="#8b8d93",
             accent="#52b788", sidebar="#15171c", ring_bg="#2a2d35",
             chart_bg="#1a1d24", chart_plot="#1e2028", chart_grid="#2a2d35",
             chart_text="#e8e6e1", chart_line="#3a3d45",
             hero="linear-gradient(135deg,#162c20,#1a4030,#234d35)",
             alert_ok_bg="#162417", alert_ok_bd="#234d2c", alert_ok_tx="#6ee7a0",
             alert_w_bg="#2a2010", alert_w_bd="#4a3a1a", alert_w_tx="#e8c35e",
             alert_c_bg="#2a1515", alert_c_bd="#4a2020", alert_c_tx="#f87171",
             env_t="#2a2218", env_h="#18292a", env_l="#2a2818", env_w="#221e2e",
             pump_on_bg="#1a3a26", pump_on_tx="#6ee7a0",
             ov_1="#18292a", ov_2="#1a2030", ov_3="#2a2218",
             hover_shadow="rgba(0,0,0,0.2)")
else:
    T = dict(bg="#f6f4ef", card="#ffffff", border="#e6e2d9", text="#1a1a1a", muted="#7a7568",
             accent="#2d6a4f", sidebar="#ffffff", ring_bg="#f0ece4",
             chart_bg="rgba(0,0,0,0)", chart_plot="#fdfcfa", chart_grid="#ede9e1",
             chart_text="#1a1a1a", chart_line="#ccc5b9",
             hero="linear-gradient(135deg,#264235,#2d6a4f,#52b788)",
             alert_ok_bg="#f0fdf4", alert_ok_bd="#bbf7d0", alert_ok_tx="#166534",
             alert_w_bg="#fff8f0", alert_w_bd="#f0d9b5", alert_w_tx="#8a6d3b",
             alert_c_bg="#fef2f2", alert_c_bd="#fecaca", alert_c_tx="#991b1b",
             env_t="#fef3e2", env_h="#e6f7f1", env_l="#fefce8", env_w="#ede9fe",
             pump_on_bg="#dcfce7", pump_on_tx="#166534",
             ov_1="#f0fdf4", ov_2="#eff6ff", ov_3="#fef3e2",
             hover_shadow="rgba(0,0,0,0.06)")


def css():
    # Use string.Template style ($key) to avoid f-string brace issues with CSS
    import string
    tpl = string.Template("""
<link rel="stylesheet" href="https://unpkg.com/@phosphor-icons/web@2.0.3/src/regular/style.css"/>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
.stApp { background:$bg !important; font-family:'DM Sans',sans-serif !important; }
header[data-testid="stHeader"] { background:transparent !important; }
section[data-testid="stSidebar"] { background:$sidebar !important; border-right:1px solid $border !important; }
section[data-testid="stSidebar"] * { color:$text !important; }
#MainMenu, footer, .stDeployButton { display:none !important; }
.stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:2px solid $border; }
.stTabs [data-baseweb="tab"] { background:transparent; border:none; padding:10px 22px; font-weight:600; color:$muted; border-radius:12px 12px 0 0; font-size:0.9rem; }
.stTabs [aria-selected="true"] { color:$accent !important; border-bottom:2px solid $accent !important; background:transparent !important; }
.hero { background:$hero; border-radius:24px; padding:40px 48px; color:#fff; margin-bottom:28px; position:relative; overflow:hidden; }
.hero::before { content:''; position:absolute; top:-60px; right:-40px; width:260px; height:260px; border-radius:50%; background:rgba(255,255,255,0.05); }
.hero::after { content:''; position:absolute; bottom:-80px; left:30%; width:400px; height:200px; border-radius:50%; background:rgba(255,255,255,0.03); }
.hero h1 { font-size:2.2rem; font-weight:800; margin:0; letter-spacing:-1px; position:relative; }
.hero .sub { font-size:0.95rem; opacity:0.8; margin:6px 0 0 0; position:relative; }
.badge-row { margin-top:16px; display:flex; gap:8px; flex-wrap:wrap; position:relative; }
.hbadge { background:rgba(255,255,255,0.12); backdrop-filter:blur(8px); padding:6px 16px; border-radius:24px; font-size:0.78rem; font-weight:600; display:inline-flex; align-items:center; gap:6px; border:1px solid rgba(255,255,255,0.1); }
.hbadge i { font-size:0.95rem; }
.hbadge.live { background:rgba(82,183,136,0.3); border-color:rgba(82,183,136,0.4); }
.alert-bar { background:$alert_w_bg; border:1px solid $alert_w_bd; border-radius:14px; padding:12px 20px; margin-bottom:20px; display:flex; align-items:center; gap:12px; font-size:0.85rem; color:$alert_w_tx; font-weight:500; }
.alert-bar i { font-size:1.3rem; color:#e8963e; }
.alert-bar.critical { background:$alert_c_bg; border-color:$alert_c_bd; color:$alert_c_tx; }
.alert-bar.critical i { color:#d44040; }
.alert-bar.ok { background:$alert_ok_bg; border-color:$alert_ok_bd; color:$alert_ok_tx; }
.alert-bar.ok i { color:#2e9e6e; }
.env-strip { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:24px; }
.env-chip { background:$card; border:1px solid $border; border-radius:16px; padding:16px; display:flex; align-items:center; gap:14px; transition:box-shadow 0.2s,transform 0.2s; }
.env-chip:hover { box-shadow:0 4px 20px $hover_shadow; transform:translateY(-2px); }
.env-ico { width:44px; height:44px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:1.2rem; }
.env-ico.t { background:$env_t; color:#e8963e; }
.env-ico.h { background:$env_h; color:#2e9e6e; }
.env-ico.l { background:$env_l; color:#ca8a04; }
.env-ico.w { background:$env_w; color:#7c5cbf; }
.env-chip h4 { margin:0; font-size:1.15rem; font-weight:700; color:$text; }
.env-chip p { margin:0; font-size:0.7rem; color:$muted; font-weight:600; text-transform:uppercase; letter-spacing:0.8px; }
.sec-header { display:flex; align-items:center; gap:10px; margin:4px 0 16px 0; }
.sec-header h3 { margin:0; font-size:1.1rem; font-weight:700; color:$text; }
.sec-header .sec-line { flex:1; height:1px; background:$border; }
.plant-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-bottom:28px; }
@media(max-width:1000px){ .plant-grid{grid-template-columns:repeat(2,1fr);} }
.pcard { background:$card; border:1px solid $border; border-radius:18px; padding:24px; transition:transform 0.2s,box-shadow 0.2s; }
.pcard:hover { transform:translateY(-4px); box-shadow:0 12px 40px $hover_shadow; }
.pcard-top { display:flex; align-items:center; justify-content:space-between; margin-bottom:20px; }
.pcard-id { display:flex; align-items:center; gap:12px; }
.pcard-avatar { width:48px; height:48px; border-radius:14px; background:$bg; display:flex; align-items:center; justify-content:center; font-size:1.7rem; }
.pcard-name { font-size:0.95rem; font-weight:700; color:$text; }
.pcard-sub { font-size:0.72rem; color:$muted; font-weight:500; }
.pump-pill { font-size:0.68rem; font-weight:700; padding:5px 14px; border-radius:20px; display:inline-flex; align-items:center; gap:5px; letter-spacing:0.5px; text-transform:uppercase; }
.pump-pill.on { background:$pump_on_bg; color:$pump_on_tx; animation:softpulse 2s infinite; }
.pump-pill.off { background:$bg; color:$muted; }
@keyframes softpulse { 0%,100%{box-shadow:0 0 0 0 rgba(46,158,110,0.2)} 50%{box-shadow:0 0 0 8px rgba(46,158,110,0)} }
.gauge-row { display:flex; align-items:center; gap:20px; margin-bottom:16px; }
.gauge-info { flex:1; }
.gauge-big { font-size:2.4rem; font-weight:800; line-height:1; }
.gauge-label { font-size:0.75rem; color:$muted; font-weight:500; margin-top:2px; }
.ring-wrap { position:relative; width:72px; height:72px; flex-shrink:0; }
.ring-wrap svg { transform:rotate(-90deg); }
.ring-val { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-size:0.7rem; font-weight:700; color:$text; }
.bar-track { width:100%; height:6px; background:$ring_bg; border-radius:3px; overflow:hidden; margin-bottom:14px; }
.bar-fill { height:100%; border-radius:3px; transition:width 0.5s; }
.pcard-meta { display:flex; justify-content:space-between; font-size:0.7rem; color:$muted; font-weight:500; }
.pcard-meta span { display:inline-flex; align-items:center; gap:4px; }
.pcard-meta i { font-size:0.8rem; }
.overview-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; margin-bottom:24px; }
.ov-card { background:$card; border:1px solid $border; border-radius:18px; padding:20px; text-align:center; }
.ov-card .ov-icon { width:44px; height:44px; border-radius:12px; display:inline-flex; align-items:center; justify-content:center; font-size:1.3rem; margin-bottom:10px; }
.ov-card .ov-val { font-size:1.8rem; font-weight:800; color:$text; }
.ov-card .ov-label { font-size:0.72rem; color:$muted; font-weight:500; margin-top:2px; }
.stats-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:24px; }
.stat-card { background:$card; border:1px solid $border; border-radius:18px; padding:20px; text-align:center; }
.stat-card .stat-icon { width:40px; height:40px; border-radius:12px; display:inline-flex; align-items:center; justify-content:center; font-size:1.2rem; margin-bottom:8px; background:$bg; color:$accent; }
.stat-card .stat-val { font-size:1.5rem; font-weight:800; color:$text; }
.stat-card .stat-label { font-size:0.73rem; color:$muted; font-weight:500; margin-top:2px; }
.tank-wrap { display:flex; align-items:center; gap:24px; }
.tank { width:64px; height:120px; border:3px solid $border; border-radius:0 0 16px 16px; position:relative; overflow:hidden; background:$bg; border-top:3px solid $border; }
.tank::before { content:''; position:absolute; top:-1px; left:-8px; right:-8px; height:6px; background:$border; border-radius:3px; }
.tank-fill { position:absolute; bottom:0; left:0; right:0; background:linear-gradient(180deg,#93e1c4,#2e9e6e); transition:height 0.6s; border-radius:0 0 13px 13px; }
.tank-fill.low { background:linear-gradient(180deg,#fecaca,#d44040); }
.tank-pct { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-size:0.85rem; font-weight:800; color:$text; z-index:1; }
.tank-info h4 { margin:0 0 4px 0; font-size:1rem; font-weight:700; color:$text; }
.tank-info p { margin:0; font-size:0.78rem; color:$muted; line-height:1.5; }
.log-box { background:$card; border:1px solid $border; border-radius:14px; padding:14px 18px; font-family:'SF Mono','Fira Code','Consolas',monospace; font-size:0.76rem; max-height:240px; overflow-y:auto; color:$text; }
.log-line { padding:4px 0; border-bottom:1px solid $border; }
.log-line:last-child { border-bottom:none; }
</style>""")
    return tpl.substitute(T)


st.markdown(css(), unsafe_allow_html=True)

# ── State init ──
if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.logs = []
    st.session_state.tick_count = 0
    st.session_state.total_water_used = 0.0
    st.session_state.pump_events = 0
    st.session_state.temp = 28.0
    st.session_state.humidity = 55.0
    st.session_state.light = 7200
    st.session_state.wind = 12.0
    st.session_state.water_tank = 100.0
    st.session_state.plants = {}
    for p in PLANTS:
        st.session_state.plants[p["id"]] = {
            "moisture": float(p["start"]), "pump_on": False,
            "history": [], "times_watered": 0, "total_water": 0.0,
        }

for key, val in [("threshold_mult", 1.0), ("drain_mult", 1.0), ("water_rate", 5.0)]:
    if key not in st.session_state:
        st.session_state[key] = val


def add_log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.insert(0, f"[{ts}] {msg}")
    st.session_state.logs = st.session_state.logs[:150]


def tick():
    s = st.session_state
    s.tick_count += 1
    now = datetime.now().strftime("%H:%M:%S")
    s.temp = max(18.0, min(42.0, s.temp + random.uniform(-0.4, 0.4)))
    s.humidity = max(20.0, min(90.0, s.humidity + random.uniform(-0.6, 0.6)))
    s.light = max(1000, min(12000, s.light + random.uniform(-100, 100)))
    s.wind = max(0.0, min(35.0, s.wind + random.uniform(-0.5, 0.5)))
    heat_factor = 1.0 + max(0, (s.temp - 30)) * 0.05
    for pdef in PLANTS:
        ps = s.plants[pdef["id"]]
        drain = pdef["drain"] * heat_factor * s.drain_mult + random.uniform(-0.4, 0.4)
        ps["moisture"] -= drain
        eff = pdef["threshold"] * s.threshold_mult
        if not ps["pump_on"] and ps["moisture"] <= eff:
            ps["pump_on"] = True
            s.pump_events += 1
            add_log(f"{pdef['emoji']} {pdef['name']} at {ps['moisture']:.1f}% — PUMP ON")
        if ps["pump_on"]:
            gain = s.water_rate + random.uniform(-0.3, 0.3)
            ps["moisture"] += gain
            ps["total_water"] += gain * 0.1
            s.total_water_used += gain * 0.1
            s.water_tank = max(0, s.water_tank - gain * 0.08)
            if ps["moisture"] >= eff + 25:
                ps["pump_on"] = False
                ps["times_watered"] += 1
                add_log(f"{pdef['emoji']} {pdef['name']} restored to {ps['moisture']:.1f}% — PUMP OFF")
        ps["moisture"] = max(0.0, min(100.0, ps["moisture"]))
        ps["history"].append({"time": now, "moisture": round(ps["moisture"], 1), "pump": ps["pump_on"]})
        if len(ps["history"]) > 200:
            ps["history"] = ps["history"][-200:]
    if not any(s.plants[p["id"]]["pump_on"] for p in PLANTS):
        s.water_tank = min(100, s.water_tank + 0.3)


def ring_svg(pct, color, size=72, stroke=6):
    r = (size - stroke) / 2
    circ = 2 * math.pi * r
    offset = circ * (1 - pct / 100)
    return (f'<div class="ring-wrap" style="width:{size}px;height:{size}px">'
            f'<svg width="{size}" height="{size}"><circle cx="{size/2}" cy="{size/2}" r="{r}" '
            f'fill="none" stroke="{T["ring_bg"]}" stroke-width="{stroke}"/>'
            f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" '
            f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
            f'stroke-dasharray="{circ}" stroke-dashoffset="{offset}"/></svg>'
            f'<div class="ring-val">{pct:.0f}%</div></div>')


CL = dict(
    template="plotly_dark" if DK else "plotly_white",
    height=420, margin=dict(l=50, r=20, t=60, b=50),
    paper_bgcolor=T["chart_bg"], plot_bgcolor=T["chart_plot"],
    font=dict(family="DM Sans,sans-serif", color=T["chart_text"], size=13),
    title_font=dict(color=T["chart_text"], size=16),
    xaxis=dict(gridcolor=T["chart_grid"], showline=True, linecolor=T["chart_line"],
               tickfont=dict(color=T["chart_text"], size=11), title_font=dict(color=T["chart_text"], size=13)),
    yaxis=dict(gridcolor=T["chart_grid"], showline=True, linecolor=T["chart_line"],
               tickfont=dict(color=T["chart_text"], size=11), title_font=dict(color=T["chart_text"], size=13)),
)

# ── Sidebar ──
with st.sidebar:
    st.toggle("Dark Mode", key="dark_mode", on_change=lambda: None)
    st.divider()
    st.markdown("### Controls")
    st.session_state.threshold_mult = st.slider("Threshold scale", 0.5, 2.0, st.session_state.threshold_mult, 0.1)
    st.session_state.drain_mult = st.slider("Evaporation speed", 0.3, 3.0, st.session_state.drain_mult, 0.1)
    st.session_state.water_rate = st.slider("Pump flow rate", 1.0, 15.0, st.session_state.water_rate, 0.5)
    st.divider()
    st.markdown("##### Plant Profiles")
    for p in PLANTS:
        eff = p["threshold"] * st.session_state.threshold_mult
        st.markdown(f"{p['emoji']} **{p['name']}** — {eff:.0f}% / {p['drain']}/t")
    st.divider()
    st.markdown("##### How it works")
    st.markdown("1. Each plant has a **soil sensor**\n2. Moisture < threshold → **pump ON**\n3. Restored → **auto OFF**\n4. Higher temp = faster drain")

# ── Hero ──
pumps_on = sum(1 for p in PLANTS if st.session_state.plants[p["id"]]["pump_on"])
live_cls = "hbadge live" if st.session_state.running else "hbadge"
st.markdown(f"""<div class="hero">
    <h1><i class="ph ph-plant" style="margin-right:10px"></i>Smart Garden IoT</h1>
    <p class="sub">Real-time multi-plant irrigation monitoring with automatic pump control</p>
    <div class="badge-row">
        <span class="{live_cls}"><i class="ph ph-pulse"></i> {"LIVE" if st.session_state.running else "PAUSED"}</span>
        <span class="hbadge"><i class="ph ph-cpu"></i> {len(PLANTS)} Sensors</span>
        <span class="hbadge"><i class="ph ph-drop"></i> {pumps_on} Pumping</span>
        <span class="hbadge"><i class="ph ph-timer"></i> Tick #{st.session_state.tick_count}</span>
        <span class="hbadge"><i class="ph ph-gas-can"></i> {st.session_state.total_water_used:.1f}L Total</span>
    </div>
</div>""", unsafe_allow_html=True)

# ── Toggle + Reset ──
c1, c2, _ = st.columns([1.2, 1, 4])
with c1:
    if st.session_state.running:
        if st.button("Stop Simulation", use_container_width=True, type="primary", icon=":material/stop:"):
            st.session_state.running = False
            for pid in st.session_state.plants:
                st.session_state.plants[pid]["pump_on"] = False
            add_log("Simulation STOPPED")
    else:
        if st.button("Start Simulation", use_container_width=True, type="primary", icon=":material/play_arrow:"):
            st.session_state.running = True
            add_log("Simulation STARTED")
with c2:
    if st.button("Reset", use_container_width=True, icon=":material/refresh:"):
        for p in PLANTS:
            st.session_state.plants[p["id"]] = {
                "moisture": float(p["start"]), "pump_on": False,
                "history": [], "times_watered": 0, "total_water": 0.0,
            }
        st.session_state.logs, st.session_state.tick_count = [], 0
        st.session_state.total_water_used, st.session_state.pump_events = 0.0, 0
        st.session_state.water_tank = 100.0
        st.session_state.running = False
        add_log("Simulation RESET")

if st.session_state.running:
    tick()

# ── Alert bar ──
critical = [p for p in PLANTS if st.session_state.plants[p["id"]]["moisture"] <= p["threshold"] * st.session_state.threshold_mult]
if critical:
    names = ", ".join(f'{p["emoji"]} {p["name"]}' for p in critical)
    cls = "alert-bar critical" if len(critical) >= 3 else "alert-bar"
    ico = "ph-warning" if len(critical) >= 3 else "ph-bell-ringing"
    st.markdown(f'<div class="{cls}"><i class="ph {ico}"></i>{names} — moisture below threshold! Pumps activated.</div>', unsafe_allow_html=True)
elif st.session_state.tick_count > 0:
    st.markdown('<div class="alert-bar ok"><i class="ph ph-check-circle"></i>All plants healthy — moisture levels normal.</div>', unsafe_allow_html=True)

# ── Overview row ──
avg_m = sum(st.session_state.plants[p["id"]]["moisture"] for p in PLANTS) / len(PLANTS)
health = min(100, max(0, avg_m * 1.2))
hcolor = "#2e9e6e" if health > 60 else "#d4920a" if health > 35 else "#d44040"
tank = st.session_state.water_tank

col_ov, col_tank = st.columns([3, 1])
with col_ov:
    st.markdown(f"""<div class="overview-grid">
        <div class="ov-card"><div class="ov-icon" style="background:{T['ov_1']};color:#2e9e6e"><i class="ph ph-heartbeat"></i></div>
            <div class="ov-val" style="color:{hcolor}">{health:.0f}</div><div class="ov-label">Garden Health Score</div></div>
        <div class="ov-card"><div class="ov-icon" style="background:{T['ov_2']};color:#3b82f6"><i class="ph ph-chart-line-up"></i></div>
            <div class="ov-val">{avg_m:.1f}%</div><div class="ov-label">Average Moisture</div></div>
        <div class="ov-card"><div class="ov-icon" style="background:{T['ov_3']};color:#e8963e"><i class="ph ph-lightning"></i></div>
            <div class="ov-val">{st.session_state.pump_events}</div><div class="ov-label">Pump Activations</div></div>
    </div>""", unsafe_allow_html=True)

with col_tank:
    tank_cls = "" if tank > 30 else " low"
    st.markdown(f"""<div class="tank-wrap" style="justify-content:center;">
        <div><div class="tank"><div class="tank-fill{tank_cls}" style="height:{tank}%"></div>
            <div class="tank-pct">{tank:.0f}%</div></div></div>
        <div class="tank-info"><h4>Water Tank</h4><p>{st.session_state.total_water_used:.1f}L used<br>{pumps_on} pumps active</p></div>
    </div>""", unsafe_allow_html=True)

# ── Env strip ──
st.markdown(f"""<div class="env-strip">
    <div class="env-chip"><div class="env-ico t"><i class="ph ph-thermometer-hot"></i></div>
        <div><h4>{st.session_state.temp:.1f}°C</h4><p>Temperature</p></div></div>
    <div class="env-chip"><div class="env-ico h"><i class="ph ph-drop-half-bottom"></i></div>
        <div><h4>{st.session_state.humidity:.1f}%</h4><p>Humidity</p></div></div>
    <div class="env-chip"><div class="env-ico l"><i class="ph ph-sun-dim"></i></div>
        <div><h4>{st.session_state.light:.0f} lx</h4><p>Light</p></div></div>
    <div class="env-chip"><div class="env-ico w"><i class="ph ph-wind"></i></div>
        <div><h4>{st.session_state.wind:.1f} km/h</h4><p>Wind</p></div></div>
</div>""", unsafe_allow_html=True)

# ── Plant cards ──
st.markdown(f'<div class="sec-header"><i class="ph ph-plant" style="font-size:1.3rem;color:{T["accent"]}"></i><h3>Plant Dashboard</h3><div class="sec-line"></div></div>', unsafe_allow_html=True)

cards = '<div class="plant-grid">'
for i, pdef in enumerate(PLANTS):
    ps = st.session_state.plants[pdef["id"]]
    m = ps["moisture"]
    eff = pdef["threshold"] * st.session_state.threshold_mult
    mcolor = "#2e9e6e" if m > eff + 15 else "#d4920a" if m > eff else "#d44040"
    pump_html = ('<span class="pump-pill on"><i class="ph ph-drop"></i> Watering</span>'
                 if ps["pump_on"] else '<span class="pump-pill off"><i class="ph ph-pause"></i> Idle</span>')
    bar_pct = max(0, min(100, m))
    ring = ring_svg(bar_pct, mcolor)
    cards += f"""<div class="pcard">
        <div class="pcard-top"><div class="pcard-id">
            <div class="pcard-avatar">{pdef['emoji']}</div>
            <div><div class="pcard-name">{pdef['name']}</div><div class="pcard-sub">Threshold: {eff:.0f}%</div></div>
        </div>{pump_html}</div>
        <div class="gauge-row">{ring}<div class="gauge-info">
            <div class="gauge-big" style="color:{mcolor}">{m:.1f}<span style="font-size:1rem;color:{T['muted']}">%</span></div>
            <div class="gauge-label">Soil Moisture</div></div></div>
        <div class="bar-track"><div class="bar-fill" style="width:{bar_pct}%;background:{mcolor}"></div></div>
        <div class="pcard-meta"><span><i class="ph ph-fire"></i>{pdef['drain']}/t</span>
            <span><i class="ph ph-arrows-clockwise"></i>{ps['times_watered']}x</span>
            <span><i class="ph ph-drop"></i>{ps['total_water']:.1f}L</span></div></div>"""
cards += '</div>'
st.markdown(cards, unsafe_allow_html=True)

# ── Charts ──
st.markdown(f'<div class="sec-header"><i class="ph ph-chart-line-up" style="font-size:1.3rem;color:{T["accent"]}"></i><h3>Analytics</h3><div class="sec-line"></div></div>', unsafe_allow_html=True)

tab_all, tab_single, tab_stats = st.tabs(["All Plants", "Individual", "Statistics"])

with tab_all:
    has_data = any(st.session_state.plants[p["id"]]["history"] for p in PLANTS)
    if has_data:
        fig = go.Figure()
        for i, pdef in enumerate(PLANTS):
            hist = st.session_state.plants[pdef["id"]]["history"]
            if hist:
                fig.add_trace(go.Scatter(
                    x=[h["time"] for h in hist], y=[h["moisture"] for h in hist],
                    mode="lines", name=f"{pdef['emoji']} {pdef['name']}",
                    line=dict(color=COLORS[i], width=2.5, shape="spline")))
        fig.add_hline(y=30 * st.session_state.threshold_mult,
            line_dash="dot", line_color="#d44040", line_width=1.5,
            annotation_text="Avg Threshold", annotation_position="top left",
            annotation_font=dict(color="#d44040", size=11))
        fig.update_layout(
            title=dict(text="<b>Soil Moisture — All Plants</b>", font=dict(size=16, color=T["chart_text"])),
            yaxis_title="Moisture (%)", yaxis_range=[0, 105],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(size=12, color=T["chart_text"]), bgcolor="rgba(0,0,0,0)"),
            **CL)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Press **Start** to begin.")

with tab_single:
    selected = st.selectbox("Select plant", [f"{p['emoji']} {p['name']}" for p in PLANTS])
    sel_idx = [f"{p['emoji']} {p['name']}" for p in PLANTS].index(selected)
    sp = PLANTS[sel_idx]
    ss = st.session_state.plants[sp["id"]]
    sh = ss["history"]
    if sh:
        times = [h["time"] for h in sh]
        moistures = [h["moisture"] for h in sh]
        pumps = [h["pump"] for h in sh]
        pc = COLORS[sel_idx]
        r, g, b = int(pc[1:3], 16), int(pc[3:5], 16), int(pc[5:7], 16)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=times, y=moistures, mode="lines", name="Moisture",
            line=dict(color=pc, width=3, shape="spline"),
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.06)"))
        eff = sp["threshold"] * st.session_state.threshold_mult
        fig2.add_hline(y=eff, line_dash="dot", line_color="#d44040", line_width=1.5,
            annotation_text=f"Threshold ({eff:.0f}%)", annotation_position="top left",
            annotation_font=dict(color="#d44040", size=11))
        pump_start = None
        for j, p in enumerate(pumps):
            if p and pump_start is None: pump_start = j
            elif not p and pump_start is not None:
                fig2.add_vrect(x0=times[pump_start], x1=times[j], fillcolor="rgba(46,158,110,0.06)",
                    line_width=0, annotation_text="watering", annotation_position="top left",
                    annotation_font=dict(size=9, color="#2e9e6e"))
                pump_start = None
        if pump_start is not None:
            fig2.add_vrect(x0=times[pump_start], x1=times[-1], fillcolor="rgba(46,158,110,0.06)",
                line_width=0, annotation_text="watering", annotation_position="top left",
                annotation_font=dict(size=9, color="#2e9e6e"))
        fig2.update_layout(
            title=dict(text=f"<b>{sp['emoji']} {sp['name']} — Detailed View</b>", font=dict(size=16, color=T["chart_text"])),
            yaxis_title="Moisture (%)", yaxis_range=[0, 105], **CL)
        st.plotly_chart(fig2, use_container_width=True)
        mc1, mc2, mc3 = st.columns(3)
        with mc1: st.metric("Current Moisture", f"{ss['moisture']:.1f}%")
        with mc2: st.metric("Times Watered", ss["times_watered"])
        with mc3: st.metric("Water Consumed", f"{ss['total_water']:.1f} L")
    else:
        st.info(f"No data for {sp['name']} yet.")

with tab_stats:
    if st.session_state.tick_count > 0:
        pumps_active = sum(1 for p in PLANTS if st.session_state.plants[p["id"]]["pump_on"])
        avg_ms = sum(st.session_state.plants[p["id"]]["moisture"] for p in PLANTS) / len(PLANTS)
        driest = min(PLANTS, key=lambda p: st.session_state.plants[p["id"]]["moisture"])
        wettest = max(PLANTS, key=lambda p: st.session_state.plants[p["id"]]["moisture"])
        st.markdown(f"""<div class="stats-grid">
            <div class="stat-card"><div class="stat-icon"><i class="ph ph-drop"></i></div>
                <div class="stat-val">{pumps_active}/{len(PLANTS)}</div><div class="stat-label">Pumps Active</div></div>
            <div class="stat-card"><div class="stat-icon"><i class="ph ph-chart-line-up"></i></div>
                <div class="stat-val">{avg_ms:.1f}%</div><div class="stat-label">Avg Moisture</div></div>
            <div class="stat-card"><div class="stat-icon"><i class="ph ph-gas-can"></i></div>
                <div class="stat-val">{st.session_state.total_water_used:.1f}L</div><div class="stat-label">Total Water</div></div>
            <div class="stat-card"><div class="stat-icon"><i class="ph ph-sun-horizon"></i></div>
                <div class="stat-val">{driest['emoji']} {driest['name']}</div><div class="stat-label">Driest ({st.session_state.plants[driest['id']]['moisture']:.1f}%)</div></div>
            <div class="stat-card"><div class="stat-icon"><i class="ph ph-flower-tulip"></i></div>
                <div class="stat-val">{wettest['emoji']} {wettest['name']}</div><div class="stat-label">Wettest ({st.session_state.plants[wettest['id']]['moisture']:.1f}%)</div></div>
            <div class="stat-card"><div class="stat-icon"><i class="ph ph-lightning"></i></div>
                <div class="stat-val">{st.session_state.pump_events}</div><div class="stat-label">Pump Activations</div></div>
        </div>""", unsafe_allow_html=True)
        fig3 = go.Figure(go.Bar(
            x=[f"{p['emoji']} {p['name']}" for p in PLANTS],
            y=[st.session_state.plants[p["id"]]["total_water"] for p in PLANTS],
            marker_color=COLORS, marker_line=dict(width=0), marker_cornerradius=8,
            text=[f"{st.session_state.plants[p['id']]['total_water']:.1f}L" for p in PLANTS],
            textposition="outside", textfont=dict(color=T["chart_text"], size=12, family="DM Sans")))
        fig3.update_layout(
            title=dict(text="<b>Water Consumption by Plant</b>", font=dict(size=16, color=T["chart_text"])),
            yaxis_title="Water (L)", **{**CL, "height": 340})
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Start the simulation to see statistics.")

# ── Event Log ──
with st.expander("Event Log", expanded=False):
    if st.session_state.logs:
        lines = "".join(f'<div class="log-line">{l}</div>' for l in st.session_state.logs)
        st.markdown(f'<div class="log-box">{lines}</div>', unsafe_allow_html=True)
    else:
        st.write("No events yet.")

# ── Auto-refresh ──
if st.session_state.running:
    time.sleep(1)
    st.rerun()
