#!/usr/bin/env python3
"""
Potato Blight Risk Dashboard - Kaggle Dataset Version
Uses CSV data similar to the Kaggle dataset structure
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random

# Set page config
st.set_page_config(
    page_title="Potato Blight Risk Dashboard",
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
.data-info {
    background: #e3f2fd;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    border-left: 4px solid #2196F3;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.current_data = None
    st.session_state.show_analysis = False
    st.session_state.show_prediction = False

def load_dataset(filepath="potato_blight_data.csv"):
    """Load potato blight dataset"""
    try:
        df = pd.read_csv(filepath)
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.month_name()
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None

def generate_more_data(existing_df, num_days=30):
    """Generate additional synthetic data matching dataset pattern"""
    # Calculate statistics from existing data
    temp_mean = existing_df['temperature'].mean()
    temp_std = existing_df['temperature'].std()
    humidity_mean = existing_df['humidity'].mean()
    humidity_std = existing_df['humidity'].std()
    
    # Generate new data
    new_data = []
    last_date = existing_df['date'].max()
    
    for day in range(1, num_days + 1):
        new_date = last_date + timedelta(days=day)
        
        # Generate weather values based on patterns
        temp = max(10, min(35, np.random.normal(temp_mean, temp_std)))
        humidity = max(30, min(100, np.random.normal(humidity_mean, humidity_std)))
        rainfall = max(0, np.random.exponential(8))
        wind_speed = max(2, min(25, np.random.normal(10, 4)))
        dew_point = temp - np.random.normal(3, 2)
        
        # Calculate blight risk (matching dataset logic)
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
        
        if rainfall > 10:
            score += 2
        elif rainfall > 5:
            score += 1
        
        if wind_speed < 10:
            score += 1
        
        risk = 'High' if score >= 5 else 'Medium' if score >= 3 else 'Low'
        
        new_data.append({
            'date': new_date,
            'temperature': round(temp, 1),
            'humidity': round(humidity, 1),
            'rainfall': round(rainfall, 1),
            'wind_speed': round(wind_speed, 1),
            'dew_point': round(dew_point, 1),
            'blight_risk': risk
        })
    
    return pd.DataFrame(new_data)

def analyze_dataset(df):
    """Perform analysis on the dataset"""
    analysis = {}
    
    # Basic statistics
    analysis['total_days'] = len(df)
    analysis['high_risk_days'] = len(df[df['blight_risk'] == 'High'])
    analysis['medium_risk_days'] = len(df[df['blight_risk'] == 'Medium'])
    analysis['low_risk_days'] = len(df[df['blight_risk'] == 'Low'])
    
    # Weather statistics
    analysis['avg_temp'] = df['temperature'].mean()
    analysis['avg_humidity'] = df['humidity'].mean()
    analysis['avg_rainfall'] = df['rainfall'].mean()
    analysis['avg_wind'] = df['wind_speed'].mean()
    
    # Risk by temperature range
    df['temp_range'] = pd.cut(df['temperature'], 
                              bins=[0, 15, 25, 35], 
                              labels=['Cool', 'Ideal', 'Warm'])
    analysis['risk_by_temp'] = df.groupby('temp_range')['blight_risk'].value_counts().unstack()
    
    # Risk by humidity range  
    df['humidity_range'] = pd.cut(df['humidity'], 
                                  bins=[0, 50, 70, 85, 100], 
                                  labels=['Dry', 'Moderate', 'Humid', 'Very Humid'])
    analysis['risk_by_humidity'] = df.groupby('humidity_range')['blight_risk'].value_counts().unstack()
    
    return analysis

def predict_risk(weather_data):
    """Predict blight risk for given weather conditions"""
    temp = weather_data['temperature']
    humidity = weather_data['humidity']
    rainfall = weather_data['rainfall']
    wind_speed = weather_data['wind_speed']
    
    score = 0
    
    # Temperature factor
    if 15 <= temp <= 25:
        score += 2
    elif 10 <= temp <= 30:
        score += 1
    
    # Humidity factor
    if humidity > 85:
        score += 3
    elif humidity > 70:
        score += 2
    elif humidity > 50:
        score += 1
    
    # Rainfall factor
    if rainfall > 10:
        score += 2
    elif rainfall > 5:
        score += 1
    
    # Wind speed factor
    if wind_speed < 10:
        score += 1
    
    # Classify risk
    if score >= 5:
        return 'High'
    elif score >= 3:
        return 'Medium'
    else:
        return 'Low'

# Title
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2.5rem;">🥔 Potato Blight Risk Analysis</h1>
    <p style="margin:0.5rem 0 0 0; font-size:1.1rem; opacity:0.9;">Kaggle Dataset Dashboard - Historical Weather & Blight Risk</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ Controls")
    
    # Load data
    if st.button("📊 Load Dataset", use_container_width=True, type="primary"):
        with st.spinner("Loading dataset..."):
            df = load_dataset()
            if df is not None:
                # Add some synthetic data to make it more interesting
                synthetic_df = generate_more_data(df, num_days=60)
                full_df = pd.concat([df, synthetic_df], ignore_index=True)
                
                st.session_state.dataset = full_df
                st.session_state.data_loaded = True
                st.session_state.current_data = full_df.iloc[-1].to_dict()  # Latest data
                st.success(f"✅ Loaded {len(full_df)} days of data!")
            else:
                st.error("❌ Failed to load dataset")
    
    if st.session_state.data_loaded:
        st.divider()
        st.toggle("📈 Show Analysis", key="show_analysis")
        st.toggle("🔮 Show Prediction Tool", key="show_prediction")
    
    st.divider()
    st.markdown("### 📚 About")
    st.markdown("""
    **Dataset Structure:**
    - 📅 Date
    - 🌡️ Temperature (°C)
    - 💧 Humidity (%)
    - 🌧️ Rainfall (mm)
    - 🌬️ Wind Speed (km/h)
    - 💧 Dew Point (°C)
    - 🔴 Blight Risk (Low/Medium/High)
    
    **Based on Kaggle Dataset:**
    Potato Leaf Disease Based on Weather Details
    """)

# Main content
if not st.session_state.data_loaded:
    st.info("📊 Click 'Load Dataset' in the sidebar to get started")
else:
    df = st.session_state.dataset
    
    # Current conditions
    st.markdown("## 📊 Current Conditions")
    
    if st.session_state.current_data:
        current = st.session_state.current_data
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            risk_class = f"risk-level risk-{current['blight_risk'].lower()}"
            emoji = '🟢' if current['blight_risk'] == 'Low' else '🟡' if current['blight_risk'] == 'Medium' else '🔴'
            st.markdown(f"""
            <div class="{risk_class}">
                <h3>🔮 Current Risk</h3>
                <div style="font-size: 2rem; font-weight: bold;">{emoji} {current['blight_risk']}</div>
                <p style="margin: 0.5rem 0; font-size: 0.8rem;">
                    {current['date'].strftime('%Y-%m-%d')}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("🌡️ Temperature", f"{current['temperature']}°C")
        with col3:
            st.metric("💧 Humidity", f"{current['humidity']}%")
        with col4:
            st.metric("🌧️ Rainfall", f"{current['rainfall']} mm")
    
    # Dataset overview
    st.markdown("## 📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📅 Total Days", len(df))
    with col2:
        high_days = len(df[df['blight_risk'] == 'High'])
        st.metric("🔴 High Risk Days", high_days)
    with col3:
        med_days = len(df[df['blight_risk'] == 'Medium'])
        st.metric("🟡 Medium Risk Days", med_days)
    with col4:
        low_days = len(df[df['blight_risk'] == 'Low'])
        st.metric("🟢 Low Risk Days", low_days)
    
    # Risk distribution chart
    st.markdown("## 📊 Risk Distribution")
    
    risk_counts = df['blight_risk'].value_counts()
    fig = px.pie(values=risk_counts.values, 
                 names=risk_counts.index,
                 title="Blight Risk Distribution",
                 color=risk_counts.index,
                 color_discrete_map={'Low': '#4CAF50', 'Medium': '#FFC107', 'High': '#F44336'},
                 hole=0.4)
    fig.update_layout(height=400)
    st.plotly_chart(fig, width='stretch')
    
    # Time series analysis
    st.markdown("## 📈 Historical Trends")
    
    fig = go.Figure()
    
    # Temperature
    fig.add_trace(go.Scatter(
        x=df['date'], 
        y=df['temperature'],
        name='Temperature (°C)',
        line=dict(color='#FF5722', width=2),
        yaxis='y'
    ))
    
    # Humidity
    fig.add_trace(go.Scatter(
        x=df['date'], 
        y=df['humidity'],
        name='Humidity (%)',
        line=dict(color='#2196F3', width=2),
        yaxis='y2'
    ))
    
    # Rainfall bars
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['rainfall'],
        name='Rainfall (mm)',
        marker=dict(color='#9C27B0', opacity=0.3),
        yaxis='y3'
    ))
    
    # Risk markers
    risk_colors = {'Low': '#4CAF50', 'Medium': '#FFC107', 'High': '#F44336'}
    for risk_level in ['High', 'Medium', 'Low']:
        risk_df = df[df['blight_risk'] == risk_level]
        fig.add_trace(go.Scatter(
            x=risk_df['date'],
            y=risk_df['temperature'],
            mode='markers',
            marker=dict(size=8, color=risk_colors[risk_level], symbol='diamond'),
            name=f'Risk: {risk_level}',
            showlegend=True
        ))
    
    fig.update_layout(
        height=500,
        hovermode='x unified',
        title='Weather Conditions & Blight Risk Over Time',
        xaxis=dict(title='Date', showgrid=True),
        yaxis=dict(title='Temperature (°C)', side='left'),
        yaxis2=dict(title='Humidity (%)', overlaying='y', side='right'),
        yaxis3=dict(title='Rainfall (mm)', overlaying='y', side='right', position=0.95),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Analysis section
    if st.session_state.show_analysis:
        st.markdown("## 🔬 Detailed Analysis")
        
        analysis = analyze_dataset(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='data-info'><h3>📊 Weather Statistics</h3></div>", unsafe_allow_html=True)
            st.write(f"**Average Temperature:** {analysis['avg_temp']:.1f}°C")
            st.write(f"**Average Humidity:** {analysis['avg_humidity']:.1f}%")
            st.write(f"**Average Rainfall:** {analysis['avg_rainfall']:.1f} mm")
            st.write(f"**Average Wind Speed:** {analysis['avg_wind']:.1f} km/h")
            
            st.markdown("### 🎯 High Risk Conditions")
            high_risk = df[df['blight_risk'] == 'High']
            st.write(f"**Avg Temp:** {high_risk['temperature'].mean():.1f}°C")
            st.write(f"**Avg Humidity:** {high_risk['humidity'].mean():.1f}%")
            st.write(f"**Avg Rainfall:** {high_risk['rainfall'].mean():.1f} mm")
        
        with col2:
            st.markdown("<div class='data-info'><h3>📊 Risk Patterns</h3></div>", unsafe_allow_html=True)
            
            # Temperature vs Risk
            fig_temp = px.box(df, x='blight_risk', y='temperature', 
                            color='blight_risk',
                            color_discrete_map={'Low': '#4CAF50', 'Medium': '#FFC107', 'High': '#F44336'})
            fig_temp.update_layout(title='Temperature Distribution by Risk Level', height=300)
            st.plotly_chart(fig_temp, width='stretch')
            
            # Humidity vs Risk
            fig_hum = px.box(df, x='blight_risk', y='humidity', 
                           color='blight_risk',
                           color_discrete_map={'Low': '#4CAF50', 'Medium': '#FFC107', 'High': '#F44336'})
            fig_hum.update_layout(title='Humidity Distribution by Risk Level', height=300)
            st.plotly_chart(fig_hum, width='stretch')
        
        # Correlation heatmap
        st.markdown("### 🔗 Feature Correlations")
        corr_features = ['temperature', 'humidity', 'rainfall', 'wind_speed', 'dew_point']
        corr_matrix = df[corr_features].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_features,
            y=corr_features,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))
        fig_corr.update_layout(title='Weather Feature Correlations', height=400)
        st.plotly_chart(fig_corr, width='stretch')
    
    # Prediction tool
    if st.session_state.show_prediction:
        st.markdown("## 🔮 Predict Blight Risk")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📝 Enter Weather Conditions")
            
            temp_input = st.number_input("Temperature (°C)", min_value=5.0, max_value=40.0, value=22.0, step=0.1)
            humidity_input = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0, value=85.0, step=0.1)
            rainfall_input = st.number_input("Rainfall (mm)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
            wind_input = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=30.0, value=8.0, step=0.1)
            
            if st.button("🔍 Predict Risk", type="primary"):
                weather_data = {
                    'temperature': temp_input,
                    'humidity': humidity_input,
                    'rainfall': rainfall_input,
                    'wind_speed': wind_input
                }
                
                prediction = predict_risk(weather_data)
                st.session_state.prediction = prediction
                st.session_state.prediction_data = weather_data
        
        with col2:
            if 'prediction' in st.session_state:
                prediction = st.session_state.prediction
                risk_class = f"risk-level risk-{prediction.lower()}"
                emoji = '🟢' if prediction == 'Low' else '🟡' if prediction == 'Medium' else '🔴'
                
                st.markdown(f"""
                <div class="{risk_class}">
                    <h2 style="margin:0;">{emoji} {prediction} Risk</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Show prevention tips
                st.markdown("### 💡 Recommendations")
                if prediction == 'High':
                    st.markdown("""
                    ❗ **Immediate Actions:**
                    - Apply fungicide immediately
                    - Remove infected plants
                    - Avoid overhead irrigation
                    - Increase plant spacing
                    - Monitor daily for symptoms
                    """)
                elif prediction == 'Medium':
                    st.markdown("""
                    ⚠️ **Preventive Actions:**
                    - Apply preventive fungicide
                    - Reduce leaf wetness
                    - Improve field drainage
                    - Monitor weather closely
                    - Prepare for outbreak
                    """)
                else:
                    st.markdown("""
                    ✅ **Maintenance:**
                    - Continue normal monitoring
                    - Maintain crop hygiene
                    - Ensure proper irrigation
                    - Keep weather records
                    - Stay vigilant
                    """)
    
    # Data table
    st.markdown("## 📋 Raw Data")
    st.dataframe(df.tail(20), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🥔 Potato Blight Risk Analysis Dashboard</p>
    <p>Based on Kaggle Dataset: Potato Leaf Disease Based on Weather Details</p>
    <p>Uses synthetic data augmentation for demonstration</p>
</div>
""", unsafe_allow_html=True)