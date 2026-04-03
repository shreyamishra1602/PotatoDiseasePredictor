#!/usr/bin/env python3
"""
Potato Blight Risk Prediction using Machine Learning
Predicts potato blight risk based on historical weather data
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os
import requests
import json

# Set page config
st.set_page_config(
    page_title="Potato Blight Risk Prediction",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.feature_importance = None

# Title
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2.5rem;">🥔 Potato Blight Risk Prediction</h1>
    <p style="margin:0.5rem 0 0 0; font-size:1.1rem; opacity:0.9;">Machine Learning for Early Blight Detection using Weather Data</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ Controls")
    
    # Data generation parameters
    st.subheader("Data Generation")
    num_samples = st.slider("Number of samples", 100, 5000, 1000, 100)
    start_date = st.date_input("Start date", datetime.now() - timedelta(days=365))
    
    # Model selection
    st.subheader("Model Selection")
    model_choice = st.selectbox(
        "Choose ML Model",
        ["Random Forest", "Gradient Boosting", "Support Vector Machine"]
    )
    
    test_size = st.slider("Test size (%)", 10, 50, 20, 5)
    random_state = st.number_input("Random state", 0, 100, 42)
    
    # Generate data button
    if st.button("🔄 Generate Synthetic Data", type="primary"):
        st.session_state.data_generated = True
        st.session_state.num_samples = num_samples
        st.session_state.start_date = start_date
    
    # Train model button
    if st.button("🚀 Train Model", type="primary"):
        st.session_state.train_model = True
    
    st.divider()
    st.markdown("### 📊 About")
    st.markdown("""
    This app predicts potato blight risk based on weather conditions:
    - **Temperature** (°C)
    - **Humidity** (%)
    - **Rainfall** (mm)
    - **Wind Speed** (km/h)
    - **Dew Point** (°C)
    
    Uses machine learning to classify risk as:
    - 🟢 Low Risk
    - 🟡 Medium Risk  
    - 🔴 High Risk
    """)

def generate_synthetic_data(num_samples, start_date):
    """Generate synthetic potato blight weather data"""
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(start_date, periods=num_samples, freq='D')
    
    # Generate weather features
    temp = np.random.normal(18, 6, num_samples)  # °C
    humidity = np.random.normal(75, 15, num_samples)  # %
    rainfall = np.random.exponential(5, num_samples)  # mm
    wind_speed = np.random.normal(12, 4, num_samples)  # km/h
    dew_point = temp - np.random.normal(3, 2, num_samples)  # °C
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'temperature': temp,
        'humidity': humidity,
        'rainfall': rainfall,
        'wind_speed': wind_speed,
        'dew_point': dew_point
    })
    
    # Calculate blight risk based on known conditions
    # High risk: high humidity + moderate temp + rainfall
    # Medium risk: moderate conditions
    # Low risk: dry, cool, or windy conditions
    
    def calculate_risk(row):
        score = 0
        
        # Temperature factor (15-25°C is ideal for blight)
        if 15 <= row['temperature'] <= 25:
            score += 2
        elif 10 <= row['temperature'] <= 30:
            score += 1
        
        # Humidity factor (high humidity favors blight)
        if row['humidity'] > 85:
            score += 3
        elif row['humidity'] > 70:
            score += 2
        elif row['humidity'] > 50:
            score += 1
        
        # Rainfall factor
        if row['rainfall'] > 10:
            score += 2
        elif row['rainfall'] > 5:
            score += 1
        
        # Wind speed factor (high wind reduces risk)
        if row['wind_speed'] > 20:
            score -= 1
        
        # Dew point factor
        if row['dew_point'] > 12:
            score += 1
        
        # Classify risk
        if score >= 5:
            return 'High'
        elif score >= 3:
            return 'Medium'
        else:
            return 'Low'
    
    data['risk_level'] = data.apply(calculate_risk, axis=1)
    
    # Add some noise
    noise_indices = np.random.choice(num_samples, size=int(num_samples*0.05), replace=False)
    data.loc[noise_indices, 'risk_level'] = np.random.choice(['Low', 'Medium', 'High'], size=len(noise_indices))
    
    return data

def train_model(data, model_type, test_size, random_state):
    """Train machine learning model"""
    
    # Prepare features and target
    X = data[['temperature', 'humidity', 'rainfall', 'wind_speed', 'dew_point']]
    y = data['risk_level']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    else:  # SVM
        model = SVC(kernel='rbf', probability=True, random_state=random_state)
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        feature_importance = None
    
    return model, scaler, accuracy, precision, recall, f1, feature_importance, X_test, y_test, y_pred

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
                'weather_desc': current['weatherDesc'][0]['value']
            }
        else:
            return None
    except:
        return None

def predict_risk(model, scaler, weather_data):
    """Predict blight risk for given weather conditions"""
    if model is None or scaler is None:
        return None
    
    # Convert to DataFrame and scale
    input_df = pd.DataFrame([weather_data])
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    return prediction, probabilities

# Main app
st.markdown("## 📈 Data Overview")

# Generate or load data
if 'data_generated' not in st.session_state or not st.session_state.data_generated:
    st.info("📊 Generate synthetic weather data using the sidebar controls")
    data = pd.DataFrame()
else:
    data = generate_synthetic_data(st.session_state.num_samples, st.session_state.start_date)
    
    # Show data preview
    st.success(f"✅ Generated {len(data)} samples of synthetic weather data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Data Preview")
        st.dataframe(data.head(10), use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Risk Distribution")
        risk_counts = data['risk_level'].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                     title="Blight Risk Distribution",
                     color=risk_counts.index,
                     color_discrete_map={'Low': '#4CAF50', 'Medium': '#FFC107', 'High': '#F44336'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics
    st.markdown("### 📊 Weather Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Avg Temperature", f"{data['temperature'].mean():.1f}°C")
    with col2:
        st.metric("Avg Humidity", f"{data['humidity'].mean():.1f}%")
    with col3:
        st.metric("Avg Rainfall", f"{data['rainfall'].mean():.1f}mm")
    with col4:
        st.metric("Avg Wind Speed", f"{data['wind_speed'].mean():.1f}km/h")
    with col5:
        st.metric("Avg Dew Point", f"{data['dew_point'].mean():.1f}°C")

# Train model if requested
if 'train_model' in st.session_state and st.session_state.train_model:
    if 'data_generated' in st.session_state and st.session_state.data_generated:
        with st.spinner("🚀 Training model..."):
            model, scaler, accuracy, precision, recall, f1, feature_importance, X_test, y_test, y_pred = train_model(
                data, model_choice, test_size, random_state
            )
            
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.model_trained = True
            st.session_state.feature_importance = feature_importance
            st.session_state.test_data = (X_test, y_test, y_pred)
            
            st.success("✅ Model trained successfully!")
    else:
        st.warning("⚠️ Please generate data first")

# Show model performance if trained
if st.session_state.model_trained:
    st.markdown("## 🤖 Model Performance")
    
    X_test, y_test, y_pred = st.session_state.test_data
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("Precision", f"{precision:.3f}")
    with col3:
        st.metric("Recall", f"{recall:.3f}")
    with col4:
        st.metric("F1 Score", f"{f1:.3f}")
    
    # Confusion matrix
    st.markdown("### 📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Low', 'Medium', 'High'],
                    y=['Low', 'Medium', 'High'],
                    color_continuous_scale='Viridis',
                    text_auto=True)
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    if st.session_state.feature_importance is not None:
        st.markdown("### 📊 Feature Importance")
        fig = px.bar(st.session_state.feature_importance, x='importance', y='feature', 
                     orientation='h', title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

# Prediction interface
st.markdown("## 🔮 Predict Blight Risk")

if st.session_state.model_trained:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📍 Current Weather (wttr.in)")
        
        location = st.text_input("Location", "Delhi")
        
        if st.button("🔄 Fetch Current Weather", type="primary"):
            weather_data = get_current_weather(location)
            if weather_data:
                st.session_state.current_weather = weather_data
                st.success(f"✅ Fetched weather for {location}")
            else:
                st.error("❌ Could not fetch weather data")
        
        st.markdown("---")
        st.markdown("### 📝 Or Enter Manual Weather Conditions")
        
        temp_input = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.0, step=0.1)
        humidity_input = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
        rainfall_input = st.number_input("Rainfall (mm)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        wind_input = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=12.0, step=0.1)
        dew_input = st.number_input("Dew Point (°C)", min_value=-10.0, max_value=40.0, value=15.0, step=0.1)
        
        if st.button("🔍 Predict Risk", type="primary"):
            weather_data = {
                'temperature': temp_input,
                'humidity': humidity_input,
                'rainfall': rainfall_input,
                'wind_speed': wind_input,
                'dew_point': dew_input
            }
            
            prediction, probabilities = predict_risk(st.session_state.model, st.session_state.scaler, weather_data)
            
            st.session_state.prediction = prediction
            st.session_state.probabilities = probabilities
    
    with col2:
        # Show current weather if available
        if 'current_weather' in st.session_state:
            weather = st.session_state.current_weather
            st.markdown(f"""
            <div class="metric-card">
                <h3>📍 {location} Weather</h3>
                <p><strong>Temperature:</strong> {weather['temperature']}°C</p>
                <p><strong>Humidity:</strong> {weather['humidity']}%</p>
                <p><strong>Wind Speed:</strong> {weather['wind_speed']} km/h</p>
                <p><strong>Conditions:</strong> {weather['weather_desc']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if 'prediction' in st.session_state:
            prediction = st.session_state.prediction
            probabilities = st.session_state.probabilities
            
            # Display prediction
            if prediction == 'Low':
                risk_class = 'risk-level risk-low'
                emoji = '🟢'
            elif prediction == 'Medium':
                risk_class = 'risk-level risk-medium' 
                emoji = '🟡'
            else:
                risk_class = 'risk-level risk-high'
                emoji = '🔴'
            
            st.markdown(f"""
            <div class="{risk_class}">
                <h2 style="margin:0;">{emoji} {prediction} Risk</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Show probabilities
            st.markdown("### 📊 Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Risk Level': ['Low', 'Medium', 'High'],
                'Probability': probabilities
            })
            
            fig = px.bar(prob_df, x='Risk Level', y='Probability', 
                        color='Risk Level',
                        color_discrete_map={'Low': '#4CAF50', 'Medium': '#FFC107', 'High': '#F44336'},
                        title="Risk Probabilities")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show prevention tips
            st.markdown("### 💡 Prevention Recommendations")
            
            if prediction == 'High':
                st.markdown("""
                ❗ **High Risk Actions:**
                - Apply fungicide immediately
                - Remove infected plants
                - Avoid overhead irrigation
                - Increase plant spacing for airflow
                - Monitor daily for symptoms
                """)
            elif prediction == 'Medium':
                st.markdown("""
                ⚠️ **Medium Risk Actions:**
                - Apply preventive fungicide
                - Reduce leaf wetness duration
                - Improve field drainage
                - Monitor weather conditions closely
                - Prepare for potential outbreak
                """)
            else:
                st.markdown("""
                ✅ **Low Risk Actions:**
                - Continue normal monitoring
                - Maintain good crop hygiene
                - Ensure proper irrigation
                - Keep records of weather conditions
                - Stay vigilant for changes
                """)

# Data visualization
if 'data_generated' in st.session_state and st.session_state.data_generated and len(data) > 0:
    st.markdown("## 📊 Advanced Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperature vs Humidity scatter
        fig = px.scatter(data, x='temperature', y='humidity', color='risk_level',
                        title='Temperature vs Humidity',
                        color_discrete_map={'Low': '#4CAF50', 'Medium': '#FFC107', 'High': '#F44336'},
                        labels={'temperature': 'Temperature (°C)', 'humidity': 'Humidity (%)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Rainfall distribution
        fig = px.histogram(data, x='rainfall', color='risk_level',
                          title='Rainfall Distribution by Risk Level',
                          barmode='overlay',
                          color_discrete_map={'Low': '#4CAF50', 'Medium': '#FFC107', 'High': '#F44336'},
                          opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series of risk
    st.markdown("### 📅 Risk Level Over Time")
    fig = px.line(data, x='date', y='risk_level', color='risk_level',
                 title='Blight Risk Over Time',
                 color_discrete_map={'Low': '#4CAF50', 'Medium': '#FFC107', 'High': '#F44336'},
                 labels={'date': 'Date', 'risk_level': 'Risk Level'})
    fig.update_yaxes(categoryorder='array', categoryarray=['Low', 'Medium', 'High'])
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🥔 Potato Blight Risk Prediction System | Machine Learning for Agriculture</p>
    <p>Uses synthetic data for demonstration purposes</p>
</div>
""", unsafe_allow_html=True)