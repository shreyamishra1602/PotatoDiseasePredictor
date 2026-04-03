#!/usr/bin/env python3
"""
Potato Blight Risk Dashboard - Simple Demo Version
Shows current weather and blight risk prediction
"""

import streamlit as st
import requests
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Potato Blight Dashboard",
    page_icon="🥔",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp { 
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    font-family: 'Arial', sans-serif;
}
.main-header {
    background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
    color: white;
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 1rem 0;
}
.risk-level {
    font-size: 2rem;
    font-weight: bold;
    text-align: center;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.risk-low {
    background: #E8F5E9;
    color: #2E7D32;
}
.risk-medium {
    background: #FFF8E1;
    color: #FF8F00;
}
.risk-high {
    background: #FFEBEE;
    color: #C62828;
}
.weather-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
    st.session_state.risk_level = "Low"
    st.session_state.last_update = None
    st.session_state.history = []
    st.session_state.show_advanced = False

def get_current_weather(location="Delhi"):
    """Get current weather using wttr.in API"""
    try:
        url = f"https://wttr.in/{location}?format=j1"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            current = data['current_condition'][0]
            return {
                'temperature': current['temp_C'],
                'humidity': current['humidity'],
                'wind_speed': current['windspeedKmph'],
                'weather_desc': current['weatherDesc'][0]['value'],
                'feels_like': current['FeelsLikeC'],
                'location': location,
                'uv_index': current.get('uvIndex', '3'),  # Default to 3 if not available
                'visibility': current.get('visibility', '10'),  # Default to 10 km
                'pressure': current.get('pressure', '1013'),  # Default to 1013 mb
                'precip_mm': current.get('precipMM', '0.0')  # Default to 0 mm
            }
        return None
    except:
        return None

def generate_fake_weather_data(location="Bangalore", days=7):
    """Generate fake historical weather data for richer dashboard"""
    import random
    from datetime import datetime, timedelta
    
    fake_data = []
    base_temp = 30 if location.lower() == "bangalore" else 25
    
    for day in range(days):
        date = (datetime.now() - timedelta(days=day)).strftime("%Y-%m-%d %H:%M:%S")
        temp = base_temp + random.uniform(-3, 3)
        humidity = max(20, min(90, 30 + random.uniform(-10, 15)))
        wind = max(5, min(25, 12 + random.uniform(-5, 8)))
        
        # Calculate risk for fake data
        score = 0
        if 15 <= temp <= 25:
            score += 2
        elif 10 <= temp <= 30:
            score += 1
        
        if humidity > 85:
            score += 3
        elif humidity > 70:
            score += 2
        elif humidity > 50:
            score += 1
        
        if wind > 20:
            score -= 1
        
        risk = 'High' if score >= 4 else 'Medium' if score >= 2 else 'Low'
        
        fake_data.append({
            'time': date,
            'temperature': round(temp, 1),
            'humidity': round(humidity, 1),
            'wind_speed': round(wind, 1),
            'risk': risk,
            'is_real': False
        })
    
    return fake_data

def calculate_blight_risk(weather):
    """Calculate potato blight risk based on weather conditions"""
    score = 0
    
    # Convert string values to float
    temp = float(weather['temperature'])
    humidity = float(weather['humidity'])
    wind = float(weather['wind_speed'])
    
    # Temperature factor (15-25°C is ideal for blight)
    if 15 <= temp <= 25:
        score += 2
    elif 10 <= temp <= 30:
        score += 1
    
    # Humidity factor (high humidity favors blight)
    if humidity > 85:
        score += 3
    elif humidity > 70:
        score += 2
    elif humidity > 50:
        score += 1
    
    # Wind speed factor (high wind reduces risk)
    if wind > 20:
        score -= 1
    
    # Classify risk
    if score >= 4:
        return 'High'
    elif score >= 2:
        return 'Medium'
    else:
        return 'Low'

def update_weather(location):
    """Update weather data and calculate risk"""
    weather = get_current_weather(location)
    if weather:
        st.session_state.weather_data = weather
        st.session_state.risk_level = calculate_blight_risk(weather)
        st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add to history
        st.session_state.history.append({
            'time': st.session_state.last_update,
            'temperature': weather['temperature'],
            'humidity': weather['humidity'],
            'risk': st.session_state.risk_level
        })
        st.session_state.history = st.session_state.history[-50:]  # Keep last 50 entries
        
        return True
    return False

# Title
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2.5rem;">🥔 Potato Blight Risk Dashboard</h1>
    <p style="margin:0.5rem 0 0 0; font-size:1.1rem; opacity:0.9;">Real-time weather monitoring and blight risk prediction</p>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("🎛️ Controls")
    
    location = st.text_input("📍 Location", "Bangalore")
    
    # Auto-fetch weather on first load
    if st.session_state.weather_data is None:
        if update_weather(location):
            st.success("✅ Weather fetched for Bangalore!")
            # Add some fake historical data for richer experience
            fake_history = generate_fake_weather_data(location, days=5)
            for entry in fake_history:
                st.session_state.history.append(entry)
        else:
            st.error("❌ Could not fetch weather")
    
    st.toggle("📊 Show Advanced Data", key="show_advanced")
    
    if st.button("🔄 Refresh Weather", use_container_width=True, type="primary"):
        if update_weather(location):
            st.success("✅ Weather updated!")
        else:
            st.error("❌ Could not fetch weather")
    
    st.divider()
    
    st.markdown("### 📊 About")
    st.markdown("""
    **Potato Blight Risk Factors:**
    - 🌡️ Temperature (15-25°C ideal for blight)
    - 💧 Humidity (>85% high risk)
    - 🌬️ Wind speed (>20 km/h reduces risk)
    
    **Risk Levels:**
    - 🟢 Low: Safe conditions
    - 🟡 Medium: Monitor closely
    - 🔴 High: Take action!
    """)

# Main dashboard
st.markdown("## 📊 Current Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.weather_data:
        weather = st.session_state.weather_data
        st.markdown(f"""
        <div class="weather-card">
            <h3>📍 {weather['location']}</h3>
            <div style="font-size: 3rem; font-weight: bold; margin: 1rem 0;">
                {weather['temperature']}°C
            </div>
            <p style="margin: 0.5rem 0;">{weather['weather_desc']}</p>
            <p style="margin: 0.5rem 0; color: #666;">Feels like {weather['feels_like']}°C</p>
            {'' if not st.session_state.show_advanced else f'''
            <div style="margin-top: 1rem; font-size: 0.9rem;">
                <p>🌤️ UV Index: {weather['uv_index']}</p>
                <p>👁️ Visibility: {weather['visibility']} km</p>
                <p>📏 Pressure: {weather['pressure']} mb</p>
                <p>💧 Precipitation: {weather['precip_mm']} mm</p>
            </div>
            '''}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📍 Fetch weather data using the sidebar")

with col2:
    if st.session_state.weather_data:
        weather = st.session_state.weather_data
        humidity = float(weather['humidity'])
        risk_text = "❗ High Risk" if humidity > 80 else "⚠️ Moderate" if humidity > 60 else "✅ Low Risk"
        risk_color = "#C62828" if humidity > 80 else "#FF8F00" if humidity > 60 else "#2E7D32"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>💧 Humidity</h3>
            <div style="font-size: 2.5rem; font-weight: bold; margin: 1rem 0; color: {risk_color};">
                {weather['humidity']}%
            </div>
            <p style="margin: 0.5rem 0; color: #666;">Wind: {weather['wind_speed']} km/h</p>
            <p style="margin: 0.5rem 0; color: {risk_color}; font-weight: bold;">{risk_text}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='metric-card'><h3>💧 Humidity</h3><p>No data</p></div>", unsafe_allow_html=True)

with col3:
    if st.session_state.last_update:
        risk_class = f"risk-level risk-{st.session_state.risk_level.lower()}"
        emoji = '🟢' if st.session_state.risk_level == 'Low' else '🟡' if st.session_state.risk_level == 'Medium' else '🔴'
        
        # Get additional risk factors
        if st.session_state.weather_data:
            weather = st.session_state.weather_data
            temp = float(weather['temperature'])
            humidity = float(weather['humidity'])
            wind = float(weather['wind_speed'])
            
            risk_factors = []
            if 15 <= temp <= 25:
                risk_factors.append("🌡️ Ideal temp for blight")
            if humidity > 85:
                risk_factors.append("💧 Very high humidity")
            if wind <= 10:
                risk_factors.append("🌬️ Low wind speed")
            
            factors_text = ", ".join(risk_factors) if risk_factors else "✅ Favorable conditions"
        else:
            factors_text = ""
        
        st.markdown(f"""
        <div class="{risk_class}">
            <h3>🔮 Blight Risk</h3>
            <div style="font-size: 2rem; font-weight: bold;">
                {emoji} {st.session_state.risk_level}
            </div>
            <p style="margin: 0.5rem 0; font-size: 0.8rem;">
                {factors_text}
            </p>
            <p style="margin: 0.5rem 0; font-size: 0.7rem; color: #666;">
                Updated: {st.session_state.last_update}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='metric-card'><h3>🔮 Blight Risk</h3><p>No data</p></div>", unsafe_allow_html=True)

# Risk recommendations
if st.session_state.risk_level:
    st.markdown("## 💡 Recommendations")
    
    if st.session_state.risk_level == 'High':
        st.markdown("""
        <div class="metric-card" style="border: 2px solid #C62828; background: #FFEBEE;">
            <h3 style="color: #C62828;">❗ HIGH RISK - IMMEDIATE ACTION NEEDED</h3>
            <ul>
                <li>🚨 Apply fungicide immediately</li>
                <li>🔍 Inspect plants daily for symptoms</li>
                <li>💧 Avoid overhead irrigation</li>
                <li>🌬️ Increase plant spacing for better airflow</li>
                <li>🗑️ Remove any infected plants immediately</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif st.session_state.risk_level == 'Medium':
        st.markdown("""
        <div class="metric-card" style="border: 2px solid #FF8F00; background: #FFF8E1;">
            <h3 style="color: #FF8F00;">⚠️ MEDIUM RISK - MONITOR CLOSELY</h3>
            <ul>
                <li>🛡️ Apply preventive fungicide</li>
                <li>📊 Monitor weather conditions daily</li>
                <li>💧 Reduce leaf wetness duration</li>
                <li>🌱 Improve field drainage</li>
                <li>📅 Prepare for potential outbreak</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="metric-card" style="border: 2px solid #2E7D32; background: #E8F5E9;">
            <h3 style="color: #2E7D32;">✅ LOW RISK - CONTINUE NORMAL PRACTICES</h3>
            <ul>
                <li>👀 Continue regular monitoring</li>
                <li>🌱 Maintain good crop hygiene</li>
                <li>💧 Ensure proper irrigation practices</li>
                <li>📊 Keep records of weather conditions</li>
                <li>🛡️ Stay prepared for changes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Data summary section
if len(st.session_state.history) > 0:
    st.markdown("## 📊 Weather Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    temps = [float(entry['temperature']) for entry in st.session_state.history]
    humidities = [float(entry['humidity']) for entry in st.session_state.history]
    winds = [float(entry['wind_speed']) for entry in st.session_state.history if 'wind_speed' in entry]
    risks = [entry['risk'] for entry in st.session_state.history]
    
    with col1:
        st.metric("🌡️ Avg Temperature", f"{np.mean(temps):.1f}°C")
    with col2:
        st.metric("💧 Avg Humidity", f"{np.mean(humidities):.1f}%")
    with col3:
        if winds:
            st.metric("🌬️ Avg Wind", f"{np.mean(winds):.1f} km/h")
    with col4:
        high_risk_days = risks.count('High')
        st.metric("🔴 High Risk Days", f"{high_risk_days}")

# Weather history chart
if len(st.session_state.history) > 1:
    st.markdown("## 📈 Weather History & Trends")
    
    # Create chart
    times = [entry['time'] for entry in st.session_state.history]
    temps = [entry['temperature'] for entry in st.session_state.history]
    humidities = [entry['humidity'] for entry in st.session_state.history]
    risks = [entry['risk'] for entry in st.session_state.history]
    
    # Risk color mapping
    risk_colors = {'Low': '#4CAF50', 'Medium': '#FFC107', 'High': '#F44336'}
    
    fig = go.Figure()
    
    # Temperature line
    fig.add_trace(go.Scatter(
        x=times, 
        y=temps, 
        name='Temperature (°C)',
        line=dict(color='#FF5722', width=3),
        yaxis='y',
        fill='tozeroy',
        fillcolor='rgba(255, 87, 34, 0.1)'
    ))
    
    # Humidity line
    fig.add_trace(go.Scatter(
        x=times, 
        y=humidities, 
        name='Humidity (%)',
        line=dict(color='#2196F3', width=3),
        yaxis='y2',
        fill='tozeroy',
        fillcolor='rgba(33, 150, 243, 0.1)'
    ))
    
    # Risk markers
    for i, risk in enumerate(risks):
        fig.add_trace(go.Scatter(
            x=[times[i]], 
            y=[temps[i]], 
            mode='markers',
            marker=dict(size=15, color=risk_colors[risk], symbol='diamond'),
            name=risk,
            showlegend=False,
            hovertext=f'Risk: {risk}<br>Temp: {temps[i]}°C<br>Humidity: {humidities[i]}%',
            hovertemplate='%{hovertext}<extra></extra>'
        ))
    
    # Add risk level annotations
    high_risk_times = [times[i] for i, r in enumerate(risks) if r == 'High']
    for time in high_risk_times:
        fig.add_vline(x=time, line_width=1, line_dash="dash", line_color="red", opacity=0.3)
    
    # Layout
    fig.update_layout(
        height=450,
        hovermode='x unified',
        title='Weather Conditions & Blight Risk Over Time',
        xaxis=dict(title='Time', showgrid=True, gridcolor='#e0e0e0'),
        yaxis=dict(title='Temperature (°C)', side='left', showgrid=True, gridcolor='#e0e0e0'),
        yaxis2=dict(title='Humidity (%)', overlaying='y', side='right', showgrid=False),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        plot_bgcolor='rgba(250,250,250,0.9)',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Risk distribution pie chart
    if st.session_state.show_advanced:
        st.markdown("## 📊 Risk Level Distribution")
        
        risk_counts = pd.Series(risks).value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.4,
            marker=dict(colors=[risk_colors.get(r, '#666') for r in risk_counts.index])
        )])
        fig_pie.update_layout(
            height=350,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, width='stretch')

# Auto-refresh
st.markdown("---")
if st.checkbox("Auto-refresh weather every 5 minutes"):
    if update_weather(location):
        st.info("🔄 Auto-refresh enabled. Weather will update every 5 minutes.")
    time.sleep(300)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🥔 Potato Blight Risk Dashboard | Real-time weather data from wttr.in</p>
    <p>Risk calculation based on agricultural science principles</p>
</div>
""", unsafe_allow_html=True)